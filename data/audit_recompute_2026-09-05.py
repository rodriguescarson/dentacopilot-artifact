"""Recompute the numbers changed by the 2026-09-05 audit (items 6, 7, 9).

Every number written into draft/paper.tex by the 2026-09-05 fix pass that
differs from the previous master is produced here and saved next to this
script as ``audit_recompute_2026-09-05.json``. The manuscript cites that file
in a LaTeX comment wherever one of these numbers is restated.

Item 6  Table tab:calibration. ECE_pre was computed on all 1,284 test
        examples while ECE_post was computed on the 642-example held-out
        half. Recompute ECE_pre on the SAME held-out half (in_calibration_dev
        == False) so both columns describe one population.
Item 7  Table tab:abstention. The published rows used the raw top-1
        probability. Recompute coverage / top-1 (covered) at tau = 0.6 on the
        temperature-scaled (calibrated) top-1 probability, on all 1,284
        examples, which is what Section 3.4 says the sweep is over.
Item 9  Exact two-sided McNemar tests for the paired n=30 contrasts
        (B1 vs M2, B1 vs M1, M5 vs M2, B1 vs M5) from correct@1 per example
        in results_matched_n30_v2/results.json, paired by (chart_id,
        prefix_len). Report the discordant counts.

Item 5  Example count: regenerate the split from cache/synth_n500_seed42.json
        with patient_split(seed=42) and count train / dev / test examples.
Item 4  The n=15 table: M5 on the 15 keys M4 ran on (filtered from the n=30
        run) and M6 from its own run directory, so every row of that table
        has a source file.

Also recomputes, as a cross-check, the numbers the audit already verified
(raw abstention rows, ECE_post, T) so a reader can see that this script
reproduces the untouched cells too.

Run:  python3 data/audit_recompute_2026-09-05.py
Deps: numpy, scipy (binomtest). Stdlib otherwise.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration import CalibrationData, expected_calibration_error  # noqa: E402

CAL = HERE / "results_full_2026-05-02T12-35-51Z" / "results_calibrated.json"
N30 = HERE / "results_matched_n30_v2" / "results.json"
N15 = HERE / "results_matched_n15" / "results.json"
OUT = HERE / "audit_recompute_2026-09-05.json"

LABELS = {"b0_bigram": "B0", "b1_tfidf_lr": "B1", "b2_xgboost": "B2",
          "b3_multitp": "B3", "m1_haiku": "M1", "m2_sonnet_cot": "M2",
          "m3_sonnet_rag": "M3", "m4_opus_cot": "M4",
          "m5_sonnet_baseline_primed": "M5", "m6_opus_baseline_primed": "M6"}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    return float(ci.low), float(ci.high)


def exact_mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact two-sided McNemar test on the discordant pairs.

    b = A correct & B wrong; c = A wrong & B correct.  Under H0 the count
    b ~ Binomial(b + c, 0.5); p is the two-sided exact binomial p-value.
    """
    b = int(np.sum((a_correct == 1) & (b_correct == 0)))
    c = int(np.sum((a_correct == 0) & (b_correct == 1)))
    n_disc = b + c
    p = float(binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue) if n_disc else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "discordant": n_disc,
            "both_correct": int(np.sum((a_correct == 1) & (b_correct == 1))),
            "both_wrong": int(np.sum((a_correct == 0) & (b_correct == 0))),
            "p_exact_two_sided": round(p, 4)}


def main() -> None:
    out: dict = {"sources": {"calibrated": str(CAL.relative_to(HERE)),
                              "n30": str(N30.relative_to(HERE)),
                              "n15": str(N15.relative_to(HERE))}}

    # ---------------- Items 6 and 7: calibration + abstention -----------------
    trials = json.loads(CAL.read_text())
    by_model: dict[str, list[dict]] = {}
    for t in trials:
        by_model.setdefault(t["model"], []).append(t)
    cal: dict[str, dict] = {}
    for m in ("b0_bigram", "b1_tfidf_lr", "b2_xgboost"):
        ents = by_model[m]
        held = [e for e in ents if not e["in_calibration_dev"]]
        raw_all = CalibrationData(np.array([e["top_k_probs"][0] for e in ents]),
                                  np.array([e["correct@1"] for e in ents]))
        raw_held = CalibrationData(np.array([e["top_k_probs"][0] for e in held]),
                                   np.array([e["correct@1"] for e in held]))
        cal_held = CalibrationData(np.array([e["top1_prob_calibrated"] for e in held]),
                                   np.array([e["correct@1"] for e in held]))
        cal_all = np.array([e["top1_prob_calibrated"] for e in ents])
        raw_all_p = raw_all.confidences
        corr_all = raw_all.correct
        tau = 0.6
        cov_raw = raw_all_p >= tau
        cov_cal = cal_all >= tau
        cal[m] = {
            "label": LABELS[m],
            "T": ents[0]["calibration_T"],
            "n_total": len(ents), "n_heldout": len(held),
            "ece_pre_all_1284_OLD_TABLE": round(expected_calibration_error(raw_all), 4),
            "ece_pre_heldout_642_NEW": round(expected_calibration_error(raw_held), 4),
            "ece_post_heldout_642": round(expected_calibration_error(cal_held), 4),
            "abstention_tau0.6_raw_OLD_TABLE": {
                "coverage": round(float(cov_raw.mean()), 4),
                "top1_overall": round(float(corr_all.mean()), 4),
                "top1_covered": round(float(corr_all[cov_raw].mean()), 4)},
            "abstention_tau0.6_calibrated_NEW": {
                "coverage": round(float(cov_cal.mean()), 4),
                "top1_overall": round(float(corr_all.mean()), 4),
                "top1_covered": round(float(corr_all[cov_cal].mean()), 4)},
        }
    out["calibration_and_abstention"] = cal

    # ---------------- Item 9: paired McNemar on the n=30 keys ----------------
    n30 = json.loads(N30.read_text())
    keyed: dict[str, dict[tuple, int]] = {}
    for t in n30:
        keyed.setdefault(t["model"], {})[(t["chart_id"], t["prefix_len"])] = int(t["correct@1"])
    keys = sorted(set.intersection(*(set(v) for v in keyed.values())))
    assert len(keys) == 30, len(keys)
    mc: dict[str, dict] = {}
    for a, b in (("b1_tfidf_lr", "m2_sonnet_cot"), ("b1_tfidf_lr", "m1_haiku"),
                 ("m5_sonnet_baseline_primed", "m2_sonnet_cot"),
                 ("b1_tfidf_lr", "m5_sonnet_baseline_primed"),
                 ("b0_bigram", "m2_sonnet_cot"), ("b3_multitp", "m2_sonnet_cot")):
        av = np.array([keyed[a][k] for k in keys]); bv = np.array([keyed[b][k] for k in keys])
        r = exact_mcnemar(av, bv)
        r.update({"a": LABELS[a], "b": LABELS[b], "n": len(keys),
                  "a_correct": int(av.sum()), "b_correct": int(bv.sum()),
                  "a_top1": round(float(av.mean()), 4), "b_top1": round(float(bv.mean()), 4)})
        mc[f"{LABELS[a]}_vs_{LABELS[b]}"] = r
    out["mcnemar_n30"] = mc
    out["clopper_pearson_n30"] = {
        LABELS[m]: {"k": int(sum(v.values())), "n": 30,
                    "ci95": [round(x, 3) for x in clopper_pearson(int(sum(v.values())), 30)]}
        for m, v in keyed.items()}

    # ---------------- Item 4 cross-check: M2 vs M4 on the same 15 keys -------
    n15 = json.loads(N15.read_text())
    k15: dict[str, dict[tuple, dict]] = {}
    for t in n15:
        k15.setdefault(t["model"], {})[(t["chart_id"], t["prefix_len"])] = t
    keys15 = sorted(set(k15["m4_opus_cot"]))
    assert len(keys15) == 15
    n15_tab = {}
    for m in ("m1_haiku", "m2_sonnet_cot", "m3_sonnet_rag", "m4_opus_cot",
              "b0_bigram", "b1_tfidf_lr", "b2_xgboost", "b3_multitp"):
        rows = [k15[m][k] for k in keys15]
        n15_tab[LABELS[m]] = {f"top{k}": round(float(np.mean([r[f"correct@{k}"] for r in rows])), 4)
                              for k in (1, 3, 5)}
        n15_tab[LABELS[m]]["k1"] = int(sum(r["correct@1"] for r in rows))
    m2 = np.array([k15["m2_sonnet_cot"][k]["correct@1"] for k in keys15])
    m4 = np.array([k15["m4_opus_cot"][k]["correct@1"] for k in keys15])
    n15_tab["M2_vs_M4_top1_mcnemar"] = exact_mcnemar(m2, m4)
    n30_by = {}
    for t in n30:
        n30_by.setdefault(t["model"], {})[(t["chart_id"], t["prefix_len"])] = t
    m5rows = [n30_by["m5_sonnet_baseline_primed"][k] for k in keys15]
    n15_tab["M5"] = {f"top{k}": round(float(np.mean([r[f"correct@{k}"] for r in m5rows])), 4) for k in (1, 3, 5)}
    n15_tab["M5"]["k1"] = int(sum(r["correct@1"] for r in m5rows))
    m6 = json.loads((HERE / "results_full_2026-05-02T19-16-39Z" / "results.json").read_text())
    m6k = {(t["chart_id"], t["prefix_len"]): t for t in m6 if t["model"] == "m6_opus_baseline_primed"}
    assert set(m6k) == set(keys15), "M6 keys differ from M4 keys"
    m6rows = [m6k[k] for k in keys15]
    n15_tab["M6"] = {f"top{k}": round(float(np.mean([r[f"correct@{k}"] for r in m6rows])), 4) for k in (1, 3, 5)}
    n15_tab["M6"]["k1"] = int(sum(r["correct@1"] for r in m6rows))
    n15_tab["M4_vs_M6_top1_mcnemar"] = exact_mcnemar(m4, np.array([r["correct@1"] for r in m6rows]))
    out["matched_n15"] = n15_tab
    out["sources"]["m6"] = "results_full_2026-05-02T19-16-39Z/results.json"

    # ---------------- Item 5: example counts from the cached corpus ----------
    import synth_charts
    from models.base import expand_charts, patient_split
    charts = synth_charts.load(HERE / "cache" / "synth_n500_seed42.json")
    tr, dv, te = patient_split(charts, seed=42)
    n_tr, n_dv, n_te = len(expand_charts(tr)), len(expand_charts(dv)), len(expand_charts(te))
    out["example_counts"] = {"charts": [len(tr), len(dv), len(te)], "train": n_tr, "dev": n_dv,
                             "test": n_te, "total": n_tr + n_dv + n_te,
                             "source": "cache/synth_n500_seed42.json + patient_split(seed=42)"}

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
