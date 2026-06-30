"""Shared scaffolding for the LLM-backed next-procedure recommenders.

Each LLM model (M1 Haiku, M2 Sonnet+CoT, M3 Sonnet+RAG) inherits from
``LLMModel`` and overrides:
  - ``CC_MODEL_ALIAS`` — which Claude Code model alias to invoke
  - ``_build_prompt`` — the prompt template

All LLM calls go through ``cc_compute.cc_call``, which routes to the local
``claude`` CLI (Claude Code subscription) and logs every call to the run's
``llm_calls.jsonl``. There is no fallback to the Anthropic API.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter
from typing import Sequence

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from cc_compute import RunLogger, cc_call  # noqa: E402
from cdt_codes import CDT_CODES, category_for  # noqa: E402

from .base import Example, Model


PROMPT_HEADER = """You are a dental decision-support model that recommends the next CDT \
procedure for a patient. You augment dentists; you never replace clinical \
judgement. Your output is consumed programmatically.

Output strict JSON of the form:
{"predictions": [{"code": "D####", "probability": <0..1>}, ...],
 "confidence": "low" | "medium" | "high",
 "abstain": false,
 "rationale": "<brief reasoning grounded in the chart, 1-2 sentences>"}
- Probabilities for the K predictions must sum to ~1.0 (within 0.05).
- Use only ADA CDT 2024 codes. Common codes for this corpus include: %CODE_PALETTE%.
- Set "abstain" to true when the chart context is genuinely insufficient
  (e.g. no relevant prior procedures, contradictory signals). Abstaining is
  preferable to confidently guessing.
- Output ONLY the JSON object. No prose, no markdown fences, no extra text.
"""


def _build_code_palette(train_examples: Sequence[Example], top_n: int = 40) -> str:
    """Return a compact string of the most-frequent CDT codes in training,
    with their human-readable shorts. Lets the model see the label space
    without flooding the prompt with the full vocabulary.
    """
    counts: Counter[str] = Counter()
    for ex in train_examples:
        counts[ex.gold] += 1
        for ev in ex.history:
            counts[ev.code] += 1

    parts: list[str] = []
    for code, _ in counts.most_common(top_n):
        info = CDT_CODES.get(code)
        if info is None:
            parts.append(code)
            continue
        parts.append(f"{code} ({info.short})")
    return ", ".join(parts)


def _format_chart(example: Example) -> str:
    history_lines: list[str] = []
    last = example.history[-10:]  # cap to recent 10 events to keep prompts short
    for ev in last:
        info = CDT_CODES.get(ev.code)
        descr = info.short if info else ""
        tooth = f" tooth {ev.tooth}" if ev.tooth is not None else ""
        history_lines.append(f"  {ev.date}: {ev.code}{tooth} ({descr})")

    mh = ", ".join(example.medical_history) if example.medical_history else "none"
    return (
        f"Age band: {example.age_band}\n"
        f"Sex: {example.sex}\n"
        f"Medical history: {mh}\n"
        f"Recent procedures (oldest to newest, up to 10):\n"
        + ("\n".join(history_lines) if history_lines else "  (none)")
    )


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    fence_re = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.MULTILINE)
    m = fence_re.search(text.strip())
    return m.group(1) if m else text


def _extract_json(text: str) -> dict | None:
    text = _strip_code_fences(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try greedy curly-brace match
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None


def _parse_predictions(
    raw_text: str,
    allowed_codes: set[str],
    top_k: int,
) -> tuple[list[tuple[str, float]], dict]:
    """Parse the LLM's JSON output into top-K (code, prob) pairs and metadata.

    On any parse failure we fall back to an empty prediction list with
    ``parse_failed=True`` in the metadata. The evaluator counts these as
    incorrect.
    """
    blob = _extract_json(raw_text)
    meta: dict = {
        "parse_failed": False,
        "confidence": None,
        "abstain": False,
        "rationale": None,
        "rejected_codes": [],
    }
    if blob is None or not isinstance(blob, dict):
        meta["parse_failed"] = True
        return [], meta

    meta["confidence"] = blob.get("confidence")
    meta["abstain"] = bool(blob.get("abstain", False))
    meta["rationale"] = blob.get("rationale")

    preds_raw = blob.get("predictions") or []
    out: list[tuple[str, float]] = []
    for entry in preds_raw:
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        prob = entry.get("probability", entry.get("prob"))
        if not isinstance(code, str) or not isinstance(prob, (int, float)):
            continue
        if code not in allowed_codes:
            meta["rejected_codes"].append(code)
            continue
        out.append((code, float(prob)))

    out.sort(key=lambda kv: kv[1], reverse=True)
    out = out[:top_k]

    # Re-normalize so probs sum to 1 over the kept top-K
    total = sum(p for _, p in out)
    if total > 0:
        out = [(c, p / total) for c, p in out]
    return out, meta


class LLMModel(Model):
    """Base class. Subclasses set CC_MODEL_ALIAS and may override _build_prompt."""

    CC_MODEL_ALIAS: str = "haiku"
    name: str = "llm_base"

    def __init__(self, run_dir: pathlib.Path | str | None = None) -> None:
        self.run_dir = pathlib.Path(run_dir) if run_dir else None
        self.logger: RunLogger | None = (
            RunLogger(self.run_dir) if self.run_dir else None
        )
        self._code_palette: str = ""
        self._allowed_codes: set[str] = set(CDT_CODES.keys())
        self._call_index: int = 0

    def attach_logger(self, run_dir: pathlib.Path) -> None:
        self.run_dir = run_dir
        self.logger = RunLogger(run_dir)

    def fit(self, train_examples: Sequence[Example]) -> None:
        # No gradient training; we only build a code palette from the train
        # split so the prompt knows the vocabulary distribution.
        self._code_palette = _build_code_palette(train_examples, top_n=40)
        for ex in train_examples:
            self._allowed_codes.add(ex.gold)
            for ev in ex.history:
                self._allowed_codes.add(ev.code)

    def _build_prompt(self, example: Example, top_k: int) -> str:
        chart = _format_chart(example)
        header = PROMPT_HEADER.replace("%CODE_PALETTE%", self._code_palette)
        return (
            header
            + "\n\nReturn the top "
            + str(top_k)
            + " predictions for this patient.\n\nChart:\n"
            + chart
            + "\n"
        )

    def predict(self, example: Example, top_k: int = 5) -> list[tuple[str, float]]:
        prompt = self._build_prompt(example, top_k)
        self._call_index += 1
        run_id = f"{self.name}-{example.chart_id}-prefix{len(example.history)}-call{self._call_index}"
        try:
            result = cc_call(
                prompt,
                model=self.CC_MODEL_ALIAS,
                logger=self.logger,
                run_id=run_id,
                timeout_s=90.0,
            )
        except RuntimeError as exc:
            if self.logger is not None:
                self.logger.append(
                    {"run_id": run_id, "error": str(exc), "model": self.CC_MODEL_ALIAS}
                )
            return []

        preds, meta = _parse_predictions(result.text, self._allowed_codes, top_k)
        self._record_meta(run_id, meta)
        return preds

    def _record_meta(self, run_id: str, meta: dict) -> None:
        if self.logger is None:
            return
        # Append a parse-metadata line alongside the call record so we can
        # later attribute parse failures and abstentions to specific calls.
        self.logger.append({"run_id": run_id, "meta": meta})


__all__ = [
    "LLMModel",
    "_parse_predictions",
    "_build_code_palette",
    "_format_chart",
]
