"""Tiny BM25 retriever over the evidence corpus.

Used by M3 (Sonnet + RAG): given a chart prefix, retrieve the top-K
evidence cards and inject them into the prompt before the LLM produces
its prediction.

Implementation is intentionally minimal — pure-Python BM25 (Robertson
1995, Lucene defaults k1=1.5, b=0.75). Twenty cards is small enough that
no embedding model is needed; lexical retrieval over CDT codes and
clinical terms gives the LLM enough signal.
"""

from __future__ import annotations

import math
import pathlib
import re
from collections import Counter
from dataclasses import dataclass

CARDS_PATH = pathlib.Path(__file__).resolve().parent / "evidence_corpus" / "cards.md"

_CARD_RE = re.compile(r"\[(E\d{3})\]\s*(.*?)(?=\n\[E\d{3}\]|\Z)", re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass(frozen=True)
class EvidenceCard:
    cid: str  # e.g. "E001"
    text: str


def load_cards(path: pathlib.Path | str = CARDS_PATH) -> list[EvidenceCard]:
    raw = pathlib.Path(path).read_text()
    cards: list[EvidenceCard] = []
    for cid, body in _CARD_RE.findall(raw):
        text = re.sub(r"\s+", " ", body).strip()
        if text:
            cards.append(EvidenceCard(cid=cid, text=text))
    return cards


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    """Pure-Python BM25 over a small corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.cards: list[EvidenceCard] = []
        self._tokens: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    def fit(self, cards: list[EvidenceCard]) -> "BM25Retriever":
        self.cards = list(cards)
        self._tokens = [_tokenize(c.text) for c in self.cards]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        n = len(self.cards)
        self._avgdl = sum(len(t) for t in self._tokens) / max(n, 1)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }
        return self

    def search(self, query: str, top_k: int = 3) -> list[tuple[EvidenceCard, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.cards:
            return []
        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._tf):
            dl = len(self._tokens[idx])
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1.0))
                score += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.cards[i], s) for i, s in scores[:top_k] if s > 0.0]


def build_query_from_chart(chart_text: str) -> str:
    """Extract a salient lexical query from a chart-formatted string.

    For our prompts, the chart already contains the most useful retrieval
    terms (CDT code names, tooth numbers, medical history). Pass the chart
    text through directly; BM25 will weight rare terms automatically.
    """
    return chart_text


__all__ = [
    "EvidenceCard",
    "BM25Retriever",
    "load_cards",
    "build_query_from_chart",
    "CARDS_PATH",
]
