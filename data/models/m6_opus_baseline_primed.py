"""M6 — Claude Opus + CoT, conditioned on the same classical prior as M5.

Tests whether Opus's added capability compounds with the prompt-conditioning
gain that M5 demonstrates. If M6 ≈ M5, capability is irrelevant once the
prior is in the prompt; if M6 > M5, capability and priming compound.
"""

from __future__ import annotations

from .m5_sonnet_baseline_primed import SonnetBaselinePrimedModel


class OpusBaselinePrimedModel(SonnetBaselinePrimedModel):
    name = "m6_opus_baseline_primed"
    CC_MODEL_ALIAS = "opus"
