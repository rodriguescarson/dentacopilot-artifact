"""B0 — Frequency bigram baseline.

For each procedure code ``c``, learn ``P(c_{t+1} | c_t)`` from training data.
At inference, given the most recent code in the history, return the top-K
codes ranked by P. If the previous code is unseen at training time, fall back
to the marginal distribution P(c_{t+1}).

Trivial to implement; useful as a sanity floor: any model that does not beat
this is not learning anything beyond procedure popularity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from .base import Example, Model


class BigramModel(Model):
    name = "b0_bigram"

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self._cond: dict[str, Counter[str]] = defaultdict(Counter)
        self._marginal: Counter[str] = Counter()
        self._vocab: list[str] = []

    def fit(self, train_examples: Sequence[Example]) -> None:
        for ex in train_examples:
            if not ex.history:
                continue
            prev = ex.history[-1].code
            self._cond[prev][ex.gold] += 1
            self._marginal[ex.gold] += 1
        self._vocab = sorted(self._marginal.keys())

    def _distribution(self, prev: str | None) -> dict[str, float]:
        if prev is not None and prev in self._cond:
            counts = self._cond[prev]
        else:
            counts = self._marginal
        if not counts:
            return {}
        # Add-α smoothing across the observed vocab so probabilities sum to 1
        total = sum(counts.values()) + self.smoothing * len(self._vocab)
        return {
            code: (counts.get(code, 0) + self.smoothing) / total
            for code in self._vocab
        }

    def predict(self, example: Example, top_k: int = 5) -> list[tuple[str, float]]:
        prev = example.history[-1].code if example.history else None
        dist = self._distribution(prev)
        return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
