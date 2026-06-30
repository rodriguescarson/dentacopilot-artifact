"""M3 — Claude Sonnet with retrieval augmentation over the evidence corpus.

Inherits from M2 (Sonnet + chain-of-thought) and adds a top-K BM25
retrieval over short clinical-practice cards in
``data/evidence_corpus/cards.md``. Retrieved cards are injected into the
prompt as ``Relevant clinical guidance:``; the model is asked to cite
specific cards (e.g. ``[E003]``) in its rationale when applicable.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from retrieval import BM25Retriever, build_query_from_chart, load_cards  # noqa: E402

from .base import Example
from .llm_base import _format_chart
from .m2_sonnet_cot import COT_PROMPT_HEADER, SonnetCoTModel


class SonnetRagModel(SonnetCoTModel):
    name = "m3_sonnet_rag"
    CC_MODEL_ALIAS = "sonnet"
    TOP_K_RETRIEVAL = 3

    def __init__(self, run_dir: pathlib.Path | str | None = None) -> None:
        super().__init__(run_dir=run_dir)
        self._retriever = BM25Retriever().fit(load_cards())

    def _build_prompt(self, example: Example, top_k: int) -> str:
        chart = _format_chart(example)
        retrieved = self._retriever.search(
            build_query_from_chart(chart), top_k=self.TOP_K_RETRIEVAL
        )
        evidence = "\n".join(f"  [{card.cid}] {card.text}" for card, _ in retrieved)
        if not evidence:
            evidence = "  (no relevant cards retrieved)"

        header = COT_PROMPT_HEADER.replace("%CODE_PALETTE%", self._code_palette)
        return (
            header
            + "\nRelevant clinical guidance (cite by [E###] in your rationale where applicable):\n"
            + evidence
            + f"\n\nReturn the top {top_k} predictions for this patient.\n\nChart:\n"
            + chart
            + "\n"
        )
