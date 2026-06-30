"""M4 — Claude Opus with the same chain-of-thought prompt as M2.

Tests whether more model capability helps on the next-procedure task.
Same prompt, same chart format, same JSON output schema as M2 — only the
model alias changes.
"""

from __future__ import annotations

from .m2_sonnet_cot import SonnetCoTModel


class OpusCoTModel(SonnetCoTModel):
    name = "m4_opus_cot"
    CC_MODEL_ALIAS = "opus"
