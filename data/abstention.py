"""Abstention rule + coverage-risk analysis for next-procedure recommenders.

The DentaCoPilot framework treats the system as decision support: when the
LLM-emitted confidence is below a threshold, or when the model itself flags
``"abstain": true``, the system declines to recommend rather than guessing.

This module provides:
  * ``coverage_risk_curve`` — sweeps thresholds and reports (coverage, risk).
  * ``select_threshold`` — pre-registered rule: smallest τ that achieves
    ≥ ``target_precision`` on covered cases, on the dev split.
  * ``apply_abstention`` — given a threshold, mark each trial as covered or
    abstained and recompute aggregate accuracy.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class CoverageRisk:
    thresholds: np.ndarray
    coverage: np.ndarray  # fraction of cases covered (not abstained)
    risk: np.ndarray      # 1 - top1_acc among covered


def coverage_risk_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_points: int = 50,
) -> CoverageRisk:
    """Sweep τ over confidences and report coverage / risk on covered cases."""
    if confidences.size == 0:
        return CoverageRisk(np.array([]), np.array([]), np.array([]))

    thresholds = np.linspace(0.0, float(confidences.max()), n_points + 1)
    cov = np.zeros_like(thresholds)
    risk = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        covered = confidences >= t
        cov[i] = covered.mean()
        if covered.any():
            risk[i] = 1.0 - correct[covered].mean()
        else:
            risk[i] = float("nan")
    return CoverageRisk(thresholds, cov, risk)


def select_threshold(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_precision: float = 0.9,
    min_coverage: float = 0.5,
) -> float:
    """Smallest τ s.t. precision on covered cases ≥ ``target_precision``,
    subject to maintaining at least ``min_coverage`` coverage. Returns 0.0
    if the constraint cannot be met (i.e. fall back to no abstention).
    """
    if confidences.size == 0:
        return 0.0
    candidates = np.unique(np.round(confidences, 4))
    candidates = np.sort(candidates)
    best = 0.0
    for t in candidates:
        covered = confidences >= t
        if covered.mean() < min_coverage:
            continue
        if not covered.any():
            continue
        precision = correct[covered].mean()
        if precision >= target_precision:
            best = float(t)
            break
    return best


def apply_abstention_to_results(
    results_path: pathlib.Path | str,
    threshold: float,
    *,
    use_calibrated: bool = False,
) -> dict:
    """Read a results.json, apply the threshold, and return a per-model summary
    of (coverage, top1_covered, top1_overall, abstain_rate)."""
    trials = json.loads(pathlib.Path(results_path).read_text())
    by_model: dict[str, list[dict]] = {}
    for ent in trials:
        by_model.setdefault(ent["model"], []).append(ent)

    summary: dict[str, dict] = {}
    for model_name, entries in by_model.items():
        confs = []
        corr = []
        explicit_abstain = []
        for e in entries:
            probs = e.get("top_k_probs") or []
            if not probs:
                explicit_abstain.append(True)
                confs.append(0.0)
                corr.append(0)
                continue
            p1 = (
                float(e.get("top1_prob_calibrated"))
                if use_calibrated and e.get("top1_prob_calibrated") is not None
                else float(probs[0])
            )
            confs.append(p1)
            corr.append(int(e["correct@1"]))
            explicit_abstain.append(False)
        confs_a = np.array(confs)
        corr_a = np.array(corr)
        ab_a = np.array(explicit_abstain)
        covered = (confs_a >= threshold) & ~ab_a
        n = len(entries)
        summary[model_name] = {
            "threshold": round(threshold, 4),
            "n_total": n,
            "coverage": round(float(covered.mean()), 4),
            "abstain_rate": round(float((~covered).mean()), 4),
            "top1_overall": round(float(corr_a.mean()), 4),
            "top1_covered": (
                round(float(corr_a[covered].mean()), 4) if covered.any() else None
            ),
        }
    return summary


__all__ = [
    "CoverageRisk",
    "coverage_risk_curve",
    "select_threshold",
    "apply_abstention_to_results",
]
