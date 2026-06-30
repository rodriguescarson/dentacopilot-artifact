"""M2 — Claude Sonnet with explicit chain-of-thought + chart-grounded rationale.

Uses the same scaffold as M1 but routes through Claude Sonnet and adds an
explicit reasoning step before the final JSON. The prompt asks the model to
*think out loud first*, then emit the JSON, then a closing ``</thinking>``
guard so we can strip the prelude reliably.
"""

from __future__ import annotations

from typing import Sequence

from .base import Example
from .llm_base import LLMModel, PROMPT_HEADER, _format_chart


COT_PROMPT_HEADER = (
    PROMPT_HEADER
    + "\n\nBefore emitting JSON, briefly think through:\n"
    + "  1. What category of procedure is most likely (diagnostic recall, "
    + "restorative, perio maintenance, ...)?\n"
    + "  2. Are there teeth in the chart that are mid-treatment "
    + "(e.g. RCT done but no crown yet)?\n"
    + "  3. Could the model legitimately abstain because the chart context "
    + "is too sparse or contradictory?\n"
    + "Wrap your reasoning in <thinking>...</thinking> and emit the JSON "
    + "immediately after the closing tag.\n"
)


class SonnetCoTModel(LLMModel):
    name = "m2_sonnet_cot"
    CC_MODEL_ALIAS = "sonnet"

    def _build_prompt(self, example: Example, top_k: int) -> str:
        chart = _format_chart(example)
        header = COT_PROMPT_HEADER.replace("%CODE_PALETTE%", self._code_palette)
        return (
            header
            + f"\nReturn the top {top_k} predictions for this patient.\n\nChart:\n"
            + chart
            + "\n"
        )
