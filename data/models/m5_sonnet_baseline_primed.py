"""M5 — Claude Sonnet + CoT, conditioned on a classical baseline's top-K shortlist.

Hypothesis: the LLM's main accuracy failure on synthetic data is picking
codes outside the high-prior set. Showing the LLM what a classical model
predicts (with probabilities) gives it strong prior information, and the
LLM's chart-grounded reasoning can refine that prior.

Implementation: M5 inherits from M2 (Sonnet+CoT). At fit time we also fit
a classical baseline (XGBoost by default) on the same training set. At
predict time, the baseline's top-10 (code, probability) pairs are
injected into the prompt; the LLM is asked to either re-rank within that
shortlist or override it (with explicit justification).

This is *not* the same as the confidence-routed hybrid (which picks one
model's full output or the other's). M5 is a single-model approach where
the LLM sees the baseline's belief and is free to use, refine, or
override it.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Sequence

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from cdt_codes import CDT_CODES  # noqa: E402

from .b1_tfidf_lr import TfidfLrModel
from .base import Example
from .llm_base import _format_chart
from .m2_sonnet_cot import SonnetCoTModel


PRIMED_PROMPT_HEADER = """You are a dental decision-support model that recommends the next CDT \
procedure for a patient. You augment dentists; you never replace clinical \
judgement. Your output is consumed programmatically.

Output strict JSON of the form:
{"predictions": [{"code": "D####", "probability": <0..1>}, ...],
 "confidence": "low" | "medium" | "high",
 "abstain": false,
 "rationale": "<brief reasoning grounded in the chart, 1-2 sentences>"}
- Probabilities for the K predictions must sum to ~1.0 (within 0.05).
- Use only ADA CDT 2024 codes. Common codes for this corpus include: %CODE_PALETTE%.
- Set "abstain" to true when the chart context is genuinely insufficient.
- Output ONLY the JSON object. No prose, no markdown fences, no extra text.

Before emitting JSON, briefly think through:
  1. What category of procedure is most likely?
  2. Are there teeth in the chart that are mid-treatment?
  3. Does the classical model's prior look right, or should you override?
Wrap your reasoning in <thinking>...</thinking> and emit JSON immediately
after the closing tag.

A separately-trained classical model (TF-IDF + multinomial logistic
regression on chart-history tokens) has predicted the following
candidates for this patient. Use this as a PRIOR --- you are free to
re-rank within the candidates, redistribute probability mass, or
override entirely if the chart strongly contradicts the model. Cite
specific candidate codes in your rationale where you agree or disagree.

Classical baseline candidates (top-10):
%BASELINE_CANDIDATES%
"""


class SonnetBaselinePrimedModel(SonnetCoTModel):
    name = "m5_sonnet_baseline_primed"
    CC_MODEL_ALIAS = "sonnet"

    def __init__(self, run_dir: pathlib.Path | str | None = None) -> None:
        super().__init__(run_dir=run_dir)
        self._baseline = TfidfLrModel()

    def fit(self, train_examples: Sequence[Example]) -> None:
        super().fit(train_examples)
        # Train the baseline on the same data
        self._baseline.fit(train_examples)

    def _format_baseline_candidates(self, example: Example) -> str:
        preds = self._baseline.predict(example, top_k=10)
        lines = []
        for code, prob in preds:
            info = CDT_CODES.get(code)
            short = f" ({info.short})" if info else ""
            lines.append(f"  {code}{short}: P={prob:.3f}")
        return "\n".join(lines) if lines else "  (no candidates)"

    def _build_prompt(self, example: Example, top_k: int) -> str:
        chart = _format_chart(example)
        candidates = self._format_baseline_candidates(example)
        header = (
            PRIMED_PROMPT_HEADER
            .replace("%CODE_PALETTE%", self._code_palette)
            .replace("%BASELINE_CANDIDATES%", candidates)
        )
        return (
            header
            + f"\nReturn the top {top_k} predictions for this patient.\n\nChart:\n"
            + chart
            + "\n"
        )
