"""Analyse the Jev Router arm (run_jev_openrouter.py) against the runs it is paired with.

Synthetic arm: results_jev_n200/results.json (M7 = Jev + the M2 chain-of-thought prompt, M8 = Jev +
the M5 prompt conditioned on B1's candidates) joined with results_stratified_n200/results.json (B0-B3,
M2, M5) on (chart_id, prefix_len). Reported per Jev design: the same fields analyze_stratified_n200
reports (top-1/3/5, exact Clopper-Pearson interval, mix-reweighted top-1, per-category top-1,
abstain and parse-failure counts), plus exact two-sided McNemar on top-1 against B1, against the
Sonnet design with the identical prompt, and between the two Jev designs; the distribution of routed
models and top-1 by routed model; billed cost and latency; and the expected calibration error of the
stated top-1 probability (10 equal-width bins), computed the same way for Sonnet for comparison.

MEPS arm: results_meps_llm_jev/results.json, Jev vs the first-order history (Markov) model on the
same 150 held-out transitions, exact McNemar, and the routed-model distribution. The Haiku probe
(21.3%) kept no per-row record, so Jev vs Haiku is reported unpaired only.

Writes analyze_jev.json next to this script; everything the manuscript restates about Jev comes
from that file.

Run:  python3 data/analyze_jev.py
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np

from analyze_stratified_n200 import cp, mcnemar

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "analyze_jev.json"
LABELS = {"b0_bigram": "B0", "b1_tfidf_lr": "B1", "b2_xgboost": "B2", "b3_multitp": "B3",
          "m2_sonnet_cot": "M2", "m5_sonnet_baseline_primed": "M5",
          "m7_jev_cot": "M7", "m8_jev_primed": "M8"}
PAIRS = [("b1_tfidf_lr", "m7_jev_cot"), ("b1_tfidf_lr", "m8_jev_primed"),
         ("m2_sonnet_cot", "m7_jev_cot"), ("m5_sonnet_baseline_primed", "m8_jev_primed"),
         ("m8_jev_primed", "m7_jev_cot"), ("b0_bigram", "m7_jev_cot")]


def ece(conf: list[float], correct: list[int], bins: int = 10) -> float:
    conf_a, corr_a = np.array(conf), np.array(correct)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf_a > lo) & (conf_a <= hi) if lo > 0 else (conf_a >= lo) & (conf_a <= hi)
        if m.any():
            total += m.mean() * abs(conf_a[m].mean() - corr_a[m].mean())
    return float(total)


def synth() -> dict:
    jev = json.loads((HERE / "results_jev_n200" / "results.json").read_text())
    base = json.loads((HERE / "results_stratified_n200" / "results.json").read_text())
    config = json.loads((HERE / "results_stratified_n200" / "config.json").read_text())
    split_counts = config["plan"]["test_split_counts"]
    split_total = sum(split_counts.values())
    by_model: dict[str, dict[tuple, dict]] = {}
    failed = Counter()
    for t in base + jev:
        if "error" in t:
            failed[t["model"]] += 1
            continue
        by_model.setdefault(t["model"], {})[(t["chart_id"], t["prefix_len"])] = t
    out: dict = {"models": {}, "failed_calls": dict(failed)}
    for m in ("b1_tfidf_lr", "m2_sonnet_cot", "m5_sonnet_baseline_primed", "m7_jev_cot", "m8_jev_primed"):
        r = list(by_model.get(m, {}).values())
        if not r:
            continue
        n, k1 = len(r), sum(t["correct@1"] for t in r)
        cats: dict[str, list[int]] = {}
        for t in r:
            cats.setdefault(t["gold_category"], []).append(t["correct@1"])
        w_acc = sum((split_counts[c] / split_total) * np.mean(v) for c, v in cats.items())
        w_cov = sum(split_counts[c] for c in cats) / split_total
        rec = {"model": m, "n": n, "k1": k1, "top1": round(k1 / n, 4),
               "top1_ci95_clopper_pearson": cp(k1, n),
               "top3": round(sum(t["correct@3"] for t in r) / n, 4),
               "top5": round(sum(t["correct@5"] for t in r) / n, 4),
               "top1_mix_reweighted": round(float(w_acc / w_cov), 4),
               "top1_by_category": {c: {"n": len(v), "top1": round(float(np.mean(v)), 4)}
                                    for c, v in sorted(cats.items())}}
        if m.startswith("m"):
            rec["abstain_count"] = sum(1 for t in r if t.get("abstain"))
            rec["parse_failed_count"] = sum(1 for t in r if t.get("parse_failed"))
            rec["empty_prediction_count"] = sum(1 for t in r if not t["top_k_codes"])
            scored = [t for t in r if t["top_k_probs"]]
            rec["ece_stated_top1_prob"] = round(ece([t["top_k_probs"][0] for t in scored],
                                                    [t["correct@1"] for t in scored]), 4)
            rec["mean_stated_top1_prob"] = round(float(np.mean([t["top_k_probs"][0] for t in scored])), 4)
            rec["confidence_labels"] = dict(Counter(str(t.get("confidence")) for t in r))
        if m.startswith(("m7", "m8")):
            routed = Counter(t.get("routed_model") for t in r)
            rec["routed_models"] = dict(routed.most_common())
            rec["top1_by_routed_model"] = {
                rm: {"n": c, "top1": round(float(np.mean([t["correct@1"] for t in r
                                                          if t.get("routed_model") == rm])), 4)}
                for rm, c in routed.most_common()}
            costs = [t.get("cost_usd") or 0.0 for t in r]
            lat = [t.get("latency_ms") for t in r if t.get("latency_ms")]
            rec["cost_usd_total"] = round(float(np.sum(costs)), 4)
            rec["cost_usd_per_call_mean"] = round(float(np.mean(costs)), 5)
            rec["latency_ms_median"] = int(np.median(lat))
            rec["latency_ms_p90"] = int(np.percentile(lat, 90))
            rec["retried_calls"] = sum(1 for t in r if (t.get("attempts") or 1) > 1)
        out["models"][LABELS[m]] = rec
    out["mcnemar_top1"] = {}
    for a, b in PAIRS:
        if a in by_model and b in by_model:
            keys = sorted(set(by_model[a]) & set(by_model[b]))
            av = np.array([by_model[a][k]["correct@1"] for k in keys])
            bv = np.array([by_model[b][k]["correct@1"] for k in keys])
            res = mcnemar(av, bv)
            res.update({"a": LABELS[a], "b": LABELS[b], "n_paired": len(keys),
                        "a_correct": int(av.sum()), "b_correct": int(bv.sum())})
            out["mcnemar_top1"][f"{LABELS[a]}_vs_{LABELS[b]}"] = res
    return out


def meps() -> dict:
    rows = json.loads((HERE / "results_meps_llm_jev" / "results.json").read_text())
    ok = [r for r in rows if "error" not in r]
    jv = np.array([r["jev_correct"] for r in ok])
    mk = np.array([r["markov_correct"] for r in ok])
    res = mcnemar(mk, jv)
    routed = Counter(r.get("routed_model") for r in ok)
    haiku = json.loads((HERE / "results_meps_llm" / "summary.json").read_text())
    return {"n": len(ok), "n_failed": len(rows) - len(ok),
            "jev_k1": int(jv.sum()), "jev_top1": round(float(jv.mean()), 4),
            "jev_top1_ci95_clopper_pearson": cp(int(jv.sum()), len(ok)),
            "markov_k1": int(mk.sum()), "markov_top1": round(float(mk.mean()), 4),
            "markov_top1_ci95_clopper_pearson": cp(int(mk.sum()), len(ok)),
            "mcnemar_markov_vs_jev": res,
            "haiku_reference_unpaired": {"n": haiku["n"], "top1": round(haiku["llm_top1"], 4),
                                         "markov_top1_reported": round(haiku["markov_top1"], 4)},
            "pred_distribution": dict(Counter(r["pred"] for r in ok).most_common()),
            "routed_models": dict(routed.most_common()),
            "cost_usd_total": round(float(sum(r.get("cost_usd") or 0 for r in ok)), 4)}


def main() -> None:
    out = {"synthetic_stratified_n200": synth(), "meps_probe_n150": meps()}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
