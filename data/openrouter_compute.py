"""OpenRouter client for the Jev Router arm (added 2026-10-10).

Every other LLM call in this project goes through ``cc_compute.cc_call`` (the
Claude Code CLI). Jev Router (TypeSafe, OpenRouter model id
``typesafe/jev-router``) is only reachable through the OpenRouter HTTP API, so
this module is the one place that talks to that API. It mirrors the
``cc_compute`` record format (run_id, model, prompt, response, tokens,
latency) and adds the fields a router makes necessary: the model OpenRouter
actually served (``routed_model``), the upstream provider, the generation id,
the billed cost, and reasoning-token counts. A router may send identical
prompts to different models, so the routed model is logged per call and the
analysis reports its distribution.

The API key is read at call time from ``OPENROUTER_API_KEY`` or, failing
that, from an ``OPENROUTER_API_KEY=`` or ``OPENROUTER_KEY=`` line of the env
file named by ``OPENROUTER_ENV_FILE``. It is never written to any log.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

API_URL = "https://openrouter.ai/api/v1/chat/completions"
JEV_MODEL = "typesafe/jev-router"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) dentacopilot-research"


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_file = os.environ.get("OPENROUTER_ENV_FILE")
    if env_file:
        for line in pathlib.Path(env_file).expanduser().read_text().splitlines():
            if line.split("=", 1)[0].strip() in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
                return line.split("=", 1)[1].strip().strip("\"'").lstrip("=")
    raise RuntimeError("no OpenRouter key: set OPENROUTER_API_KEY or OPENROUTER_ENV_FILE")


@dataclass
class ORResult:
    text: str
    routed_model: str | None
    provider: str | None
    generation_id: str | None
    finish_reason: str | None
    tokens_in: int | None
    tokens_out: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    latency_ms: int


class JsonlLog:
    """Thread-safe append-only JSONL log (one record per call)."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def or_call(prompt: str, *, model: str = JEV_MODEL, run_id: str, log: JsonlLog | None = None,
            max_tokens: int = 8000, timeout_s: float = 240.0) -> ORResult:
    """One chat completion. Raises RuntimeError on HTTP or transport failure."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "usage": {"include": True}}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
                 "User-Agent": _UA, "X-Title": "DentaCoPilot research"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:  # timeouts, resets
        raise RuntimeError(f"transport: {exc}") from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if "error" in payload:
        raise RuntimeError(f"api error: {json.dumps(payload['error'])[:300]}")
    choice = (payload.get("choices") or [{}])[0]
    usage = payload.get("usage") or {}
    res = ORResult(
        text=(choice.get("message") or {}).get("content") or "",
        routed_model=payload.get("model"), provider=payload.get("provider"),
        generation_id=payload.get("id"), finish_reason=choice.get("finish_reason"),
        tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        cost_usd=usage.get("cost"), latency_ms=latency_ms)
    if log is not None:
        log.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": run_id,
                    "requested_model": model, "routed_model": res.routed_model,
                    "provider": res.provider, "generation_id": res.generation_id,
                    "finish_reason": res.finish_reason, "prompt": prompt, "response": res.text,
                    "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                    "reasoning_tokens": res.reasoning_tokens, "cost_usd": res.cost_usd,
                    "latency_ms": latency_ms})
    return res
