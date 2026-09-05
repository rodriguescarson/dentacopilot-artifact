"""Analyse the stratified n=200 re-run (rerun_stratified_n200.py).

Reads results_stratified_n200/results.json (or, with --partial, the baselines in results.json plus
the LLM trials accumulated so far in trials_partial.jsonl) and writes analyze_stratified_n200.json
next to this script. Everything the manuscript restates from the re-run comes from that file.

Reported per model: n scored, top-1 / top-3 / top-5, exact 95% Clopper-Pearson interval on top-1,
top-1 by gold CDT category, and a mix-reweighted top-1 (post-stratified to the category mix of the
full 1,284-example test split, since the stratified draw deliberately over-samples rare
categories). Paired contrasts: exact two-sided McNemar on top-1 for B1 vs M2, B1 vs M5, M5 vs M2,
and each remaining baseline vs M2, on the keys both models scored. The n=30 numbers from
results_matched_n30_v2/summary.json are printed alongside for comparison.

Run:  python3 data/analyze_stratified_n200.py [--partial]
Deps: numpy, scipy.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / "results_stratified_n200"
OUT = HERE / "analyze_stratified_n200.json"
N30 = HERE / "results_matched_n30_v2" / "summary.json"
LABELS = {"b0_bigram": "B0", "b1_tfidf_lr": "B1", "b2_xgboost": "B2", "b3_multitp": "B3",
          "m2_sonnet_cot": "M2", "m5_sonnet_baseline_primed": "M5"}
PAIRS = [("b1_tfidf_lr", "m2_sonnet_cot"), ("b1_tfidf_lr", "m5_sonnet_baseline_primed"),
         ("m5_sonnet_baseline_primed", "m2_sonnet_cot"), ("b0_bigram", "m2_sonnet_cot"),
         ("b2_xgboost", "m2_sonnet_cot"), ("b3_multitp", "m2_sonnet_cot"),
         ("b0_bigram", "m5_sonnet_baseline_primed")]


def cp(k: int, n: int) -> list[float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return [round(ci.low, 4), round(ci.high, 4)]


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    x = int(np.sum((a == 1) & (b == 0)))
    y = int(np.sum((a == 0) & (b == 1)))
    p = float(binomtest(x, x + y, 0.5).pvalue) if x + y else 1.0
    return {"a_only_correct": x, "b_only_correct": y, "discordant": x + y,
            "both_correct": int(np.sum((a == 1) & (b == 1))), "both_wrong": int(np.sum((a == 0) & (b == 0))),
            "p_exact_two_sided": round(p, 4)}


def load(partial: bool) -> tuple[list[dict], dict]:
    config = json.loads((RUN / "config.json").read_text())
    trials = json.loads((RUN / "results.json").read_text())
    if partial:
        trials = [t for t in trials if not t["model"].startswith("m")]
        part = RUN / "trials_partial.jsonl"
        if part.exists():
            trials += [json.loads(l) for l in part.read_text().splitlines() if l.strip()]
    return trials, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="use trials_partial.jsonl for the LLM trials")
    args = ap.parse_args()
    trials, config = load(args.partial)
    trials = [t for t in trials if "error" not in t]
    failed = sum(1 for t in json.loads((RUN / "results.json").read_text()) if "error" in t)
    split_counts = config["plan"]["test_split_counts"]
    split_total = sum(split_counts.values())
    by_model: dict[str, dict[tuple, dict]] = {}
    for t in trials:
        by_model.setdefault(t["model"], {})[(t["chart_id"], t["prefix_len"])] = t

    out: dict = {"source": str(RUN.relative_to(HERE)), "partial": args.partial,
                 "sample_plan": config["plan"], "n_failed_llm_trials_in_results_json": failed, "models": {}}
    for m, rows in sorted(by_model.items()):
        r = list(rows.values())
        n = len(r)
        k1 = sum(t["correct@1"] for t in r)
        cats: dict[str, list[int]] = {}
        for t in r:
            cats.setdefault(t["gold_category"], []).append(t["correct@1"])
        # post-stratify to the test-split category mix (weight = split share / sample share)
        w_acc = sum((split_counts[c] / split_total) * np.mean(v) for c, v in cats.items())
        w_cov = sum(split_counts[c] for c in cats) / split_total
        out["models"][LABELS.get(m, m)] = {
            "model": m, "n": n, "k1": k1,
            "top1": round(k1 / n, 4), "top1_ci95_clopper_pearson": cp(k1, n),
            "top3": round(sum(t["correct@3"] for t in r) / n, 4),
            "top5": round(sum(t["correct@5"] for t in r) / n, 4),
            "top1_mix_reweighted": round(float(w_acc / w_cov), 4),
            "top1_by_category": {c: {"n": len(v), "top1": round(float(np.mean(v)), 4)} for c, v in sorted(cats.items())},
            "abstain_count": sum(1 for t in r if t.get("abstain")) if m.startswith("m") else None,
            "parse_failed_count": sum(1 for t in r if t.get("parse_failed")) if m.startswith("m") else None,
        }
    out["mcnemar_top1"] = {}
    for a, b in PAIRS:
        if a in by_model and b in by_model:
            keys = sorted(set(by_model[a]) & set(by_model[b]))
            av = np.array([by_model[a][k]["correct@1"] for k in keys])
            bv = np.array([by_model[b][k]["correct@1"] for k in keys])
            r = mcnemar(av, bv)
            r.update({"a": LABELS[a], "b": LABELS[b], "n_paired": len(keys),
                      "a_correct": int(av.sum()), "b_correct": int(bv.sum())})
            out["mcnemar_top1"][f"{LABELS[a]}_vs_{LABELS[b]}"] = r
    # n=30 reference for the same models
    n30 = json.loads(N30.read_text())
    ref = {}
    for s in n30.get("runs", n30 if isinstance(n30, list) else []):
        if s.get("model") in LABELS:
            ref[LABELS[s["model"]]] = {k: s.get(k) for k in ("n_test_examples", "top1_acc", "top3_acc", "top5_acc")}
    out["n30_reference"] = ref
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "sample_plan"}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
