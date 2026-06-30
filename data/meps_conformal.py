"""Split-conformal selective prediction on the MEPS 2023 real-data benchmark.

Strengthens the abstention story with a distribution-free guarantee. The synthetic
OOD test (abstention_ood.py) shows the abstention rule fires; here we show that on
REAL data the same confidence signal supports selective prediction with calibrated
coverage: split-conformal thresholding of the classical model's top-1 probability
yields an accepted fraction that matches the target within finite-sample tolerance,
and selective risk (error among accepted) falls monotonically as coverage tightens.

Method (split conformal, one-sided, higher score = more confident = keep):
  * Fit B2 (multinomial logistic regression) on the patient-level TRAIN split,
    exactly as meps_benchmark.py.
  * Split the held-out TEST set patient-level into CALIBRATION and EVALUATION
    (no patient in both), preserving exchangeability.
  * Score = top-1 predicted probability. For a target coverage c, set the accept
    threshold tau to the conformal (1-c) quantile of calibration scores using the
    finite-sample level ceil((1-c)(n_cal+1))/n_cal. Accept eval points with
    score >= tau.
  * Report achieved coverage (with a bootstrap 95% CI over eval patients) against
    target, the selective risk among accepted, and the model's ECE on eval.

Run: python3 meps_conformal.py
"""
from __future__ import annotations
import json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from meps_benchmark import DTA, SEED, build_transitions

import pandas as pd


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(conf)
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_stata(DTA, convert_categoricals=False)
    tr = build_transitions(df)

    # patient-level train/test split (identical to meps_benchmark)
    pids = tr["pid"].drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
    cut = int(0.7 * len(pids))
    train_pids = set(pids[:cut])
    train = tr[tr["pid"].isin(train_pids)]
    test = tr[~tr["pid"].isin(train_pids)].copy()

    # fit B2 logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr = enc.fit_transform(train[["prev_cat", "prev_prov"]])
    Xte = enc.transform(test[["prev_cat", "prev_prov"]])
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(Xtr, train["next_cat"])
    classes = clf.classes_
    proba = clf.predict_proba(Xte)
    top1_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    pred = classes[top1_idx]
    correct = (pred == test["next_cat"].values).astype(int)

    # patient-level calibration/evaluation split of the TEST set
    test_pids = test["pid"].drop_duplicates().sample(frac=1.0, random_state=SEED + 1).tolist()
    half = len(test_pids) // 2
    cal_pids = set(test_pids[:half])
    is_cal = test["pid"].isin(cal_pids).values
    s_cal, c_cal = conf[is_cal], correct[is_cal]
    s_ev, c_ev, p_ev = conf[~is_cal], correct[~is_cal], test["pid"].values[~is_cal]
    n_cal = len(s_cal)

    full_risk = float(1.0 - c_ev.mean())

    # precompute patient -> eval indices once (cluster bootstrap is over patients)
    _uniq = np.unique(p_ev)
    _pid_to_idx = {u: np.where(p_ev == u)[0] for u in _uniq}

    def boot_coverage(mask: np.ndarray, B: int = 1000) -> tuple:
        accs = np.empty(B)
        for b in range(B):
            samp = rng.choice(_uniq, size=len(_uniq), replace=True)
            idx = np.concatenate([_pid_to_idx[u] for u in samp])
            accs[b] = mask[idx].mean()
        lo, hi = np.percentile(accs, [2.5, 97.5])
        return float(lo), float(hi)

    rows = []
    for c in [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
        # conformal one-sided threshold: keep scores >= tau, target accept fraction c
        # level for the (1-c) lower quantile with finite-sample correction
        q = np.ceil((1 - c) * (n_cal + 1)) / n_cal
        q = min(max(q, 0.0), 1.0)
        tau = float(np.quantile(s_cal, q, method="lower"))
        acc = (s_ev >= tau)
        cov = float(acc.mean())
        risk = float(1.0 - c_ev[acc].mean()) if acc.sum() else float("nan")
        lo, hi = boot_coverage(acc.astype(float))
        rows.append({"target_coverage": c, "tau": tau, "achieved_coverage": cov,
                     "coverage_ci95": [lo, hi], "selective_risk": risk,
                     "n_accepted": int(acc.sum())})

    out = {
        "dataset": "MEPS HC-248B (2023 Dental Visits)", "seed": SEED,
        "model": "B2 multinomial logistic regression",
        "n_test": int(len(test)), "n_cal": int(n_cal), "n_eval": int(len(s_ev)),
        "n_eval_patients": int(len(np.unique(p_ev))),
        "full_coverage_risk": full_risk,
        "ece_eval": ece(s_ev, c_ev),
        "method": "split-conformal selective prediction, score = top-1 probability",
        "coverage_risk": rows,
    }
    outdir = HERE / "results_meps_2023"
    outdir.mkdir(exist_ok=True)
    (outdir / "conformal.json").write_text(json.dumps(out, indent=2))

    print("=== MEPS split-conformal selective prediction (B2 logreg) ===")
    print(f"n_cal={n_cal}  n_eval={len(s_ev)} ({out['n_eval_patients']} patients)")
    print(f"full coverage risk (no abstention) = {full_risk*100:.1f}%   ECE = {out['ece_eval']:.3f}\n")
    print(f"{'target':>7}{'tau':>7}{'achieved':>10}{'cov 95% CI':>18}{'sel.risk':>10}")
    for r in rows:
        ci = f"[{r['coverage_ci95'][0]*100:.1f},{r['coverage_ci95'][1]*100:.1f}]"
        sr = "nan" if r["selective_risk"] != r["selective_risk"] else f"{r['selective_risk']*100:.1f}%"
        print(f"{r['target_coverage']*100:>6.0f}%{r['tau']:>7.3f}{r['achieved_coverage']*100:>9.1f}%{ci:>18}{sr:>10}")
    print(f"\nwrote {outdir/'conformal.json'}")


if __name__ == "__main__":
    main()
