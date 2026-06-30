"""Base interface for next-procedure prediction models.

Every model in `models/` (B0–B3 baselines, M1–M3 LLM variants) implements the
same minimal interface so the evaluation harness can treat them uniformly.

A *prediction example* is the (chart, prefix_length) pair: predict the
procedure at index ``prefix_length`` given the history truncated at that
index. We expand a chart of length L into L-1 prediction examples (skipping
the very first event because there is no chart context to use).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

from synth_charts import Chart, ProcedureEvent


@dataclasses.dataclass
class Example:
    chart_id: str
    age_band: str
    sex: str
    medical_history: list[str]
    history: list[ProcedureEvent]  # truncated; gold = next event after this
    gold: str
    gold_tooth: int | None
    archetype: str | None = None


def expand_charts(charts: Iterable[Chart], min_prefix: int = 1) -> list[Example]:
    """Expand each chart into one example per (prefix_length >= min_prefix)."""
    out: list[Example] = []
    for chart in charts:
        events = chart.history
        for i in range(min_prefix, len(events)):
            out.append(
                Example(
                    chart_id=chart.patient_id,
                    age_band=chart.age_band,
                    sex=chart.sex,
                    medical_history=list(chart.medical_history),
                    history=list(events[:i]),
                    gold=events[i].code,
                    gold_tooth=events[i].tooth,
                    archetype=chart.archetype,
                )
            )
    return out


def patient_split(
    charts: list[Chart],
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Chart], list[Chart], list[Chart]]:
    """Patient-level train/dev/test split. No chart appears in two splits."""
    import random as _r

    indices = list(range(len(charts)))
    _r.Random(seed).shuffle(indices)

    n_train = int(len(charts) * train_frac)
    n_dev = int(len(charts) * dev_frac)

    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    return (
        [charts[i] for i in train_idx],
        [charts[i] for i in dev_idx],
        [charts[i] for i in test_idx],
    )


class Model:
    """Minimal model interface. Override fit() and predict()."""

    name: str = "abstract"

    def fit(self, train_examples: Sequence[Example]) -> None:
        raise NotImplementedError

    def predict(self, example: Example, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top-K (code, probability) sorted descending by probability."""
        raise NotImplementedError
