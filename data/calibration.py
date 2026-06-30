"""Calibration utilities for next-procedure prediction.

Implements:
  * Temperature scaling (Guo et al., ICML 2017): a single scalar T that
    rescales the model's top-1 probabilities to better match empirical
    accuracy. Fit by minimizing NLL on a held-out dev split.
  * Expected Calibration Error (ECE) over equal-mass bins.
  * Reliability-diagram support arrays (bin centres + accuracy + confidence).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class CalibrationData:
    confidences: np.ndarray  # shape (N,) — top-1 probability claimed by model
    correct: np.ndarray      # shape (N,) — 1 if top-1 prediction was correct


def from_results_jsonl(per_trial: Iterable[dict]) -> CalibrationData:
    """Build a CalibrationData from per-trial results.json entries."""
    confs: list[float] = []
    corr: list[int] = []
    for ent in per_trial:
        probs = ent.get("top_k_probs") or []
        if not probs:
            continue
        confs.append(float(probs[0]))
        corr.append(int(ent.get("correct@1", 0)))
    return CalibrationData(
        confidences=np.array(confs, dtype=np.float64),
        correct=np.array(corr, dtype=np.int64),
    )


def expected_calibration_error(
    data: CalibrationData,
    n_bins: int = 10,
    *,
    equal_mass: bool = False,
) -> float:
    """ECE: mean |accuracy − confidence| weighted by bin frequency."""
    if data.confidences.size == 0:
        return float("nan")
    if equal_mass:
        edges = np.quantile(data.confidences, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(data.confidences, edges[1:-1], right=False)
    n = data.confidences.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_conf = data.confidences[mask].mean()
        bin_acc = data.correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    data: CalibrationData, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_confidence_mean, bin_accuracy, bin_count)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(data.confidences, edges[1:-1], right=False)
    confs = np.zeros(n_bins)
    accs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b]:
            confs[b] = data.confidences[mask].mean()
            accs[b] = data.correct[mask].mean()
    return confs, accs, counts


def fit_temperature(
    data: CalibrationData,
    *,
    eps: float = 1e-12,
) -> float:
    """Fit a temperature T so the rescaled confidence min-NLL on dev.

    Top-1-only single-scalar formulation: we treat the prediction as a
    Bernoulli (correct vs incorrect) and rescale via
        p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T)).
    Returns the optimal T via 1-D grid search; sufficient for diagnostic use.
    """
    if data.confidences.size == 0:
        return 1.0
    p = np.clip(data.confidences, eps, 1.0 - eps)
    q = 1.0 - p
    y = data.correct.astype(np.float64)

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        a = p ** (1.0 / T)
        b = q ** (1.0 / T)
        p_t = a / (a + b)
        return -float(np.sum(y * np.log(p_t + eps) + (1.0 - y) * np.log(1.0 - p_t + eps)))

    grid = np.concatenate(
        [np.linspace(0.1, 1.0, 19), np.linspace(1.05, 5.0, 80)]
    )
    losses = [nll(t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def apply_temperature(p: float, T: float, *, eps: float = 1e-12) -> float:
    p = max(min(p, 1.0 - eps), eps)
    a = p ** (1.0 / T)
    b = (1.0 - p) ** (1.0 / T)
    return a / (a + b)


def calibrate_results_jsonl(
    in_path: pathlib.Path | str,
    out_path: pathlib.Path | str,
    *,
    dev_fraction: float = 0.5,
    seed: int = 42,
) -> dict:
    """Fit T on a random half of the trials, apply to the other half, and
    write augmented per-trial entries to ``out_path``. Return a summary dict.
    """
    in_path = pathlib.Path(in_path)
    out_path = pathlib.Path(out_path)
    trials = json.loads(in_path.read_text())

    by_model: dict[str, list[dict]] = {}
    for ent in trials:
        by_model.setdefault(ent["model"], []).append(ent)

    rng = np.random.default_rng(seed)
    augmented: list[dict] = []
    summary: dict[str, dict] = {}
    for model_name, entries in by_model.items():
        order = np.arange(len(entries))
        rng.shuffle(order)
        n_dev = int(len(entries) * dev_fraction)
        dev_idx = set(order[:n_dev].tolist())
        dev_data = from_results_jsonl(entries[i] for i in dev_idx)
        T = fit_temperature(dev_data)

        ece_pre = expected_calibration_error(from_results_jsonl(entries))
        for j, ent in enumerate(entries):
            calibrated = ent.copy()
            top_probs = ent.get("top_k_probs") or []
            if top_probs:
                p1 = float(top_probs[0])
                calibrated["top1_prob_calibrated"] = round(apply_temperature(p1, T), 6)
            calibrated["calibration_T"] = round(T, 4)
            calibrated["in_calibration_dev"] = j in dev_idx
            augmented.append(calibrated)

        # Report ECE on the test half (j not in dev_idx) using calibrated probs
        test_data = CalibrationData(
            confidences=np.array(
                [
                    apply_temperature(float(e.get("top_k_probs", [0.0])[0]), T)
                    for j, e in enumerate(entries)
                    if j not in dev_idx and e.get("top_k_probs")
                ]
            ),
            correct=np.array(
                [
                    int(e["correct@1"])
                    for j, e in enumerate(entries)
                    if j not in dev_idx and e.get("top_k_probs")
                ]
            ),
        )
        ece_post = expected_calibration_error(test_data)
        summary[model_name] = {
            "T": round(T, 4),
            "n_total": len(entries),
            "n_dev": n_dev,
            "n_test": len(entries) - n_dev,
            "ece_pre": round(ece_pre, 4),
            "ece_post": round(ece_post, 4),
        }

    out_path.write_text(json.dumps(augmented, indent=2))
    return summary


VERBALIZED_TO_PROB: dict[str, float] = {
    "low": 0.30,
    "medium": 0.60,
    "high": 0.90,
}


def map_verbalized_confidence(label: str | None, default: float = 0.5) -> float:
    """Map ``"low"|"medium"|"high"`` to a numerical confidence.

    Used for LLMs that emit a categorical confidence instead of (or alongside)
    a top-1 probability. Calibration on these labels is empirical: we learn
    a per-bucket reliability on dev and report it.
    """
    if label is None:
        return default
    return VERBALIZED_TO_PROB.get(label.strip().lower(), default)


def calibrate_verbalized(
    pairs: list[tuple[str, int]],
) -> dict[str, dict[str, float]]:
    """Given (label, correct) pairs, compute empirical accuracy per bucket.

    Returns a mapping: ``{"low": {"n": …, "acc": …}, …}``. This is the
    reliability that the paper reports for verbalized confidence.
    """
    buckets: dict[str, list[int]] = {"low": [], "medium": [], "high": []}
    for label, correct in pairs:
        if label is None:
            continue
        key = label.strip().lower()
        if key in buckets:
            buckets[key].append(int(correct))
    return {
        k: {
            "n": len(v),
            "acc": (sum(v) / len(v)) if v else float("nan"),
        }
        for k, v in buckets.items()
    }


__all__ = [
    "CalibrationData",
    "from_results_jsonl",
    "expected_calibration_error",
    "reliability_curve",
    "fit_temperature",
    "apply_temperature",
    "calibrate_results_jsonl",
    "map_verbalized_confidence",
    "calibrate_verbalized",
    "VERBALIZED_TO_PROB",
]
