"""Claude Code compute wrapper for Paper 7 (DentaCoPilot).

Every LLM call in this paper goes through this module. Calls are routed to the
local ``claude`` CLI (Claude Code), so they bill to a Claude
Code subscription rather than to a separate metered Anthropic API account.

Usage:

    from cc_compute import cc_call, RunLogger
    logger = RunLogger("results_smoke_2026-05-02T12-00-00Z")
    result = cc_call(
        "What is 2+2?",
        model="haiku",
        logger=logger,
        run_id="trial-001",
    )
    print(result.text)

Design notes:
- No ``import anthropic`` anywhere in this codebase. There is a CI grep guard
  in ``Makefile`` (``check-no-anthropic``).
- Every call is logged with prompt, response, model, latency, and token counts
  (when available) to the run's ``llm_calls.jsonl``.
- If the ``claude`` CLI is not on PATH or returns non-zero, raise. Do NOT fall
  back to a paid API.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import pathlib
import shutil
import subprocess
import time
import uuid
from typing import Any

CLAUDE_CLI = shutil.which("claude")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


@dataclasses.dataclass
class CCResult:
    text: str
    model: str
    latency_ms: float
    raw: dict[str, Any]
    run_id: str

    @property
    def tokens_in(self) -> int | None:
        usage = self.raw.get("usage") or self.raw.get("message", {}).get("usage")
        if isinstance(usage, dict):
            return usage.get("input_tokens")
        return None

    @property
    def tokens_out(self) -> int | None:
        usage = self.raw.get("usage") or self.raw.get("message", {}).get("usage")
        if isinstance(usage, dict):
            return usage.get("output_tokens")
        return None


class RunLogger:
    """One logger per experimental run; all calls append to llm_calls.jsonl."""

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.run_dir = pathlib.Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "llm_calls.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_model(model: str | None) -> str:
    if model is None:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(model, model)


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_text(blob: dict[str, Any]) -> str:
    """Best-effort extraction of the assistant response from Claude Code's
    JSON output. The exact shape varies across CLI versions; we look in
    several known places before falling back to the raw string.
    """
    if "result" in blob and isinstance(blob["result"], str):
        return blob["result"]
    if "text" in blob and isinstance(blob["text"], str):
        return blob["text"]
    msg = blob.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
            joined = "".join(chunks).strip()
            if joined:
                return joined
        if isinstance(content, str):
            return content
    return json.dumps(blob)


def cc_call(
    prompt: str,
    *,
    model: str | None = None,
    logger: RunLogger | None = None,
    run_id: str | None = None,
    timeout_s: float = 120.0,
    extra_args: list[str] | None = None,
) -> CCResult:
    """Call Claude Code with ``prompt`` and return the response.

    Parameters
    ----------
    prompt : str
        The full prompt to send. Caller is responsible for templating.
    model : str | None
        Either a friendly alias (``haiku`` / ``sonnet`` / ``opus``) or a full
        model id. Defaults to Haiku.
    logger : RunLogger | None
        If provided, the call is appended to ``<run_dir>/llm_calls.jsonl``.
    run_id : str | None
        Caller-supplied identifier (e.g. trial id). Auto-generated if omitted.
    timeout_s : float
        Hard timeout for the subprocess.
    extra_args : list[str] | None
        Extra flags forwarded to the ``claude`` CLI.

    Raises
    ------
    RuntimeError
        If the ``claude`` CLI is not on PATH, the call times out, or the CLI
        returns non-zero. We never silently fall back to a paid API.
    """
    if CLAUDE_CLI is None:
        raise RuntimeError(
            "The 'claude' CLI is not on PATH. Install Claude Code; we will not "
            "fall back to the paid Anthropic API."
        )

    resolved_model = _resolve_model(model)
    rid = run_id or f"call-{uuid.uuid4().hex[:12]}"

    cmd = [
        CLAUDE_CLI,
        "-p",
        prompt,
        "--model",
        resolved_model,
        "--output-format",
        "json",
    ]
    if extra_args:
        cmd.extend(extra_args)

    started = time.perf_counter()
    started_ts = _utcnow_iso()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        if logger is not None:
            logger.append(
                {
                    "ts": started_ts,
                    "run_id": rid,
                    "model": resolved_model,
                    "prompt": prompt,
                    "response": None,
                    "latency_ms": latency_ms,
                    "error": "timeout",
                    "tokens_in": None,
                    "tokens_out": None,
                }
            )
        raise RuntimeError(f"claude CLI timed out after {timeout_s}s") from exc

    latency_ms = (time.perf_counter() - started) * 1000.0

    if proc.returncode != 0:
        if logger is not None:
            logger.append(
                {
                    "ts": started_ts,
                    "run_id": rid,
                    "model": resolved_model,
                    "prompt": prompt,
                    "response": None,
                    "latency_ms": latency_ms,
                    "error": f"exit-{proc.returncode}",
                    "stderr": proc.stderr[:2000],
                    "tokens_in": None,
                    "tokens_out": None,
                }
            )
        raise RuntimeError(
            f"claude CLI exited with code {proc.returncode}.\nstderr:\n{proc.stderr}"
        )

    try:
        blob = json.loads(proc.stdout)
    except json.JSONDecodeError:
        blob = {"text": proc.stdout, "raw_stdout": True}

    text = _extract_text(blob)
    result = CCResult(
        text=text,
        model=resolved_model,
        latency_ms=latency_ms,
        raw=blob,
        run_id=rid,
    )

    if logger is not None:
        logger.append(
            {
                "ts": started_ts,
                "run_id": rid,
                "model": resolved_model,
                "prompt": prompt,
                "response": text,
                "latency_ms": latency_ms,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
            }
        )

    return result


__all__ = ["cc_call", "RunLogger", "CCResult", "MODEL_ALIASES", "DEFAULT_MODEL"]
