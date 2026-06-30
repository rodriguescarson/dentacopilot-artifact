"""M1 — Claude Haiku next-procedure recommender, with calibrated abstention.

Uses the shared LLMModel scaffold. Compute is routed through Claude Code
(``cc_compute.cc_call``); no Anthropic API call is ever issued.
"""

from __future__ import annotations

from .llm_base import LLMModel


class HaikuModel(LLMModel):
    name = "m1_haiku"
    CC_MODEL_ALIAS = "haiku"
