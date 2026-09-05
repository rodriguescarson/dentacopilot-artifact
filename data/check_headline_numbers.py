"""Diff every restated headline number in the manuscripts against its source file.

Item 20 of AUDIT-2026-09-05.md: after the fix pass, every number the paper restates must match the
raw result files (section 2 of the audit lists what was verified). This script loads those files,
formats each value the way the manuscript writes it (one or more accepted spellings), and reports
whether the string occurs in the master (draft/paper.tex) and in the Journal of Medical Systems
build (draft-jms/paper.tex + draft-jms/supplement.tex, searched together because the supplement
carries the moved blocks). It also asserts that a list of STALE strings (values the audit replaced)
no longer occurs anywhere.

Run:  python3 data/check_headline_numbers.py      (exit 1 on any missing or stale string)
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parent
DOCS = {
    "master": [PAPER / "draft" / "paper.tex"],
    "jms": [PAPER / "draft-jms" / "paper.tex", PAPER / "draft-jms" / "supplement.tex"],
}


def load(p: str) -> dict:
    return json.loads((HERE / p).read_text())


def strip_comments(t: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", t)


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def main() -> int:
    meps = load("results_meps_2023/summary.json")
    cif = load("results_meps_2023/ci_fairness.json")
    conf = load("results_meps_2023/conformal.json")
    ood = load("results_meps_2023/abstention_ood.json")
    fid = load("results_meps_2023/fidelity.json")
    probe = load("results_meps_llm/summary.json")
    aud = load("audit_recompute_2026-09-05.json")
    cal_trials = load("results_full_2026-05-02T12-35-51Z/results_calibrated.json")
    n30 = {r["model"]: r for r in load("results_matched_n30_v2/summary.json")["runs"]}
    full = {r["model"]: r for r in load("results_full_2026-05-02T12-35-51Z/summary.json")["runs"]}

    checks: list[tuple[str, str, list[str]]] = []   # (description, source, accepted spellings)

    def add(desc: str, src: str, *forms: str) -> None:
        checks.append((desc, src, list(forms)))

    # ---- MEPS benchmark --------------------------------------------------------------------
    r = meps["results"]
    add("MEPS transitions", "meps summary", f"{meps['n_transitions']:,}".replace(",", "{,}"))
    add("MEPS patients", "meps summary", f"{meps['n_patients']:,}".replace(",", "{,}"))
    add("MEPS test transitions", "meps summary", f"{meps['n_test']:,}".replace(",", "{,}"))
    for m, lab in (("B0_marginal", "B0"), ("B1_markov", "B1"), ("B2_logreg", "B2")):
        add(f"MEPS {lab} top-1", "meps summary", f"{r[m]['top1']:.3f}")
        add(f"MEPS {lab} top-3", "meps summary", f"{r[m]['top3']:.3f}")
        add(f"MEPS {lab} macro-F1", "meps summary", f"{r[m]['macro_f1']:.3f}")
    b1 = cif["b1"]
    add("MEPS B1 top-1 %", "ci_fairness", pct(b1["top1"]) + "\\%")
    add("MEPS B1 top-1 CI", "ci_fairness", f"[{pct(b1['top1_ci'][0])}, {pct(b1['top1_ci'][1])}]",
        f"{b1['top1_ci'][0]:.3f} to {b1['top1_ci'][1]:.3f}")
    add("MEPS B1 top-3 CI", "ci_fairness", f"[{pct(b1['top3_ci'][0])}, {pct(b1['top3_ci'][1])}]")
    add("MEPS B1 macro-F1 CI", "ci_fairness", f"[{b1['macro_f1_ci'][0]:.2f}, {b1['macro_f1_ci'][1]:.2f}]")
    for dim, (hi, lo) in {"Sex": ("Female", "Male"), "Age band": ("0-17", "65+"),
                          "Race/ethnicity": ("NH Black", "NH White"), "Poverty": ("Middle-income", "Near-poor")}.items():
        g = cif["fairness"][dim]
        add(f"subgroup {dim} max", "ci_fairness", pct(g["groups"][hi][0]) + "\\%")
        add(f"subgroup {dim} min", "ci_fairness", pct(g["groups"][lo][0]) + "\\%")
        add(f"subgroup {dim} gap", "ci_fairness", f"{pct(g['max_min_gap'])}$~pp", f"{pct(g['max_min_gap'])}~pp")
        add(f"subgroup {dim} gap CI", "ci_fairness",
            f"[{pct(g['max_min_gap_ci'][0])}, {pct(g['max_min_gap_ci'][1])}]",
            f"[{pct(g['max_min_gap_ci'][0])},\n{pct(g['max_min_gap_ci'][1])}]")
    # ---- conformal ------------------------------------------------------------------------
    add("conformal n_eval", "conformal", f"{conf['n_eval']:,}".replace(",", "{,}"))
    add("conformal patients", "conformal", str(conf["n_eval_patients"]))
    add("conformal ECE", "conformal", f"{conf['ece_eval']:.3f}")
    add("conformal full-coverage risk", "conformal", f"{conf['full_coverage_risk']:.3f}", pct(conf["full_coverage_risk"]) + "\\%")
    for row in conf["coverage_risk"]:
        if row["target_coverage"] == 0.95:
            continue
        add(f"conformal achieved cov @{row['target_coverage']}", "conformal", f"{row['achieved_coverage']:.3f}")
        add(f"conformal CI @{row['target_coverage']}", "conformal",
            f"[{row['coverage_ci95'][0]:.3f}, {row['coverage_ci95'][1]:.3f}]")
        add(f"conformal risk @{row['target_coverage']}", "conformal", f"{row['selective_risk']:.3f}")
    add("conformal risk @0.5 as %", "conformal", pct(conf["coverage_risk"][-1]["selective_risk"]) + "\\%")
    add("conformal cov @0.5 as %", "conformal", pct(conf["coverage_risk"][-1]["achieved_coverage"]) + "\\%")
    # ---- LLM probe, fidelity, synthetic abstention/OOD --------------------------------------
    add("MEPS LLM probe n", "meps_llm", str(probe["n"]))
    add("MEPS LLM probe top-1", "meps_llm", pct(probe["llm_top1"]) + "\\%")
    add("MEPS LLM probe Markov top-1", "meps_llm", pct(probe["markov_top1"]) + "\\%")
    add("fidelity marginal r", "fidelity", f"r=0.{round(fid['marginal_pearson_r'] * 100):02d}")
    add("fidelity TVD", "fidelity", f"{fid['marginal_tvd']:.2f}")
    add("fidelity transition r", "fidelity", f"r=0.{round(fid['transition_pearson_r'] * 100):02d}")
    add("synthetic tau", "abstention_ood", f"{ood['tau']:.2f}")
    add("synthetic risk no abstain", "abstention_ood", pct(ood["indist"]["risk_no_abstain"]) + "\\%", f"{ood['indist']['risk_no_abstain']:.3f}")
    add("synthetic risk covered", "abstention_ood", pct(ood["indist"]["risk_covered"]) + "\\%", f"{ood['indist']['risk_covered']:.3f}")
    for fam, lab in (("random", "random"), ("scramble", "scrambled"), ("empty", "emptied")):
        o = ood["ood"][fam]
        add(f"OOD {lab} abstain rate", "abstention_ood", f"{o['abstain_rate']:.3f}", pct(o["abstain_rate"]) + "\\%")
        add(f"OOD {lab} AUROC", "abstention_ood", f"{o['ood_auroc']:.3f}")
    half = [c for c in ood["coverage_risk_curve"] if abs(c["coverage"] - 0.5) < 0.02]
    if half:
        add("synthetic risk at 50% coverage", "abstention_ood", pct(half[0]["risk_covered"]) + "\\%")
    # ---- audit recompute (items 5-7, 9) -----------------------------------------------------
    ec = aud["example_counts"]
    for k in ("total", "train", "dev", "test"):
        add(f"example count {k}", "audit_recompute", f"{ec[k]:,}".replace(",", "{,}"))
    for m, c in aud["calibration_and_abstention"].items():
        lab = c["label"]
        add(f"{lab} T", "audit_recompute", f"{c['T']:.2f}")
        add(f"{lab} ECE pre (held-out)", "audit_recompute", f"{c['ece_pre_heldout_642_NEW']:.3f}")
        add(f"{lab} ECE post", "audit_recompute", f"{c['ece_post_heldout_642']:.3f}")
        ents = [e for e in cal_trials if e["model"] == m]
        pc = np.array([e["top1_prob_calibrated"] for e in ents]); cc = np.array([e["correct@1"] for e in ents])
        cov = pc >= 0.6
        a = {"coverage": cov.mean(), "top1_overall": cc.mean(), "top1_covered": cc[cov].mean()}
        add(f"{lab} abstention row (calibrated)", "results_calibrated.json (exact)",
            f"& {a['coverage']:.2f} & {a['top1_overall']:.3f} & {a['top1_covered']:.3f}",
            f"& {a['coverage']:.2f} & {a['top1_overall']:.3f} & \\textbf{{{a['top1_covered']:.3f}}}")
    for key, r in aud["mcnemar_n30"].items():
        if key in ("B1_vs_M2", "B1_vs_M1", "M5_vs_M2"):
            add(f"McNemar {key} p", "audit_recompute", f"p{{=}}{r['p_exact_two_sided']:.2f}", f"p{{=}}{r['p_exact_two_sided']:.3f}")
            add(f"McNemar {key} discordant", "audit_recompute", f"{r['discordant']} discordant")
    for lab, r in aud["clopper_pearson_n30"].items():
        ci = binomtest(r["k"], r["n"]).proportion_ci(confidence_level=0.95, method="exact")
        add(f"n=30 {lab} CI", "exact Clopper-Pearson", f"[{ci.low:.2f},{ci.high:.2f}]",
            f"{ci.low:.3f} to {ci.high:.3f}", f"[{ci.low:.3f}, {ci.high:.3f}]")
    for lab, r in aud["matched_n15"].items():
        if isinstance(r, dict) and "top1" in r:
            add(f"n=15 {lab} row", "audit_recompute", f"& {r['top1']:.3f} & {r['top3']:.3f} & {r['top5']:.3f}",
                f"& \\textbf{{{r['top1']:.3f}}} & {r['top3']:.3f} & \\textbf{{{r['top5']:.3f}}}",
                f"& {r['top1']:.3f} & \\textbf{{{r['top3']:.3f}}} & {r['top5']:.3f}")
    # ---- n=30 and full-split tables ----------------------------------------------------------
    for m, r in n30.items():
        add(f"n=30 {m} top-1", "n30 summary", f"{r['top1_acc']:.3f}")
        add(f"n=30 {m} top-3", "n30 summary", f"{r['top3_acc']:.3f}")
        add(f"n=30 {m} top-5", "n30 summary", f"{r['top5_acc']:.3f}")
    for m, r in full.items():
        add(f"full-split {m} row", "results_full summary",
            f"{r['top1_acc']:.3f} & {r['top3_acc']:.3f} & {r['top5_acc']:.3f}",
            f"\\textbf{{{r['top1_acc']:.3f}}} & {r['top3_acc']:.3f} & {r['top5_acc']:.3f}",
            f"{r['top1_acc']:.3f} & \\textbf{{{r['top3_acc']:.3f}}} & \\textbf{{{r['top5_acc']:.3f}}}")

    # ---- stratified n=200 re-run (item 15) ---------------------------------------------------
    rr = load("analyze_stratified_n200.json")
    for lab, v in rr["models"].items():
        add(f"n=200 {lab} top-1", "analyze_stratified_n200", f"{v['top1']:.3f}")
        add(f"n=200 {lab} top-3", "analyze_stratified_n200", f"{v['top3']:.3f}")
        add(f"n=200 {lab} top-5", "analyze_stratified_n200", f"{v['top5']:.3f}")
        ci = binomtest(v["k1"], v["n"]).proportion_ci(confidence_level=0.95, method="exact")
        add(f"n=200 {lab} CI", "exact Clopper-Pearson", f"[{ci.low:.2f},{ci.high:.2f}]")
        if lab in ("B1", "M2", "M5"):
            add(f"n=200 {lab} count", "analyze_stratified_n200", f"{v['k1']}/200")
    for key in ("B1_vs_M2", "M5_vs_M2", "B1_vs_M5", "B0_vs_M5"):
        r = rr["mcnemar_top1"][key]; pv = r["p_exact_two_sided"]
        add(f"n=200 McNemar {key} p", "analyze_stratified_n200",
            f"p{{=}}{pv:.4f}", f"p{{=}}{pv:.3f}", f"p{{=}}{pv:.2f}")
        add(f"n=200 McNemar {key} discordant", "analyze_stratified_n200",
            f"{r['a_only_correct']} {r['a']}-only", f"{r['a_only_correct']} won by {r['a']}",
            f"({r['a_only_correct']} vs {r['b_only_correct']})")
    stale = ["7{,}162", "& 1.15 & 0.054 &", "& 1.75 & 0.124 &", "0.36 & 0.558 & 0.718", "0.51 & 0.507 & 0.654",
             "identically to M2", "match M2 (Sonnet alone) on every metric", "p \\approx",
             "pre-registered KLE", "was pre-registered", "pre-registered clinician",
             "[46.2, 49.6]", "before any test-split unblinding", "JMIRx Med", "final-year BDS candidate",
             "KLE Vishwanath Katti Institute", "Sonnet to Opus does not help", "& 0.556 & 0.778 &",
             "adequately powered re-run", "200-example follow-up", "next planned run",
             "pending the stratified re-run"]

    texts = {k: "\n".join(strip_comments(p.read_text()) for p in v) for k, v in DOCS.items()}
    missing = 0
    print(f"{'check':46} {'source':18} master  jms")
    for desc, src, forms in checks:
        found = {k: any(f in t for f in forms) for k, t in texts.items()}
        flag = "" if all(found.values()) else "   <-- MISSING"
        if flag:
            missing += 1
        print(f"{desc:46} {src:18} {'yes' if found['master'] else 'NO ':6}  {'yes' if found['jms'] else 'NO '}{flag}"
              + ("" if all(found.values()) else f"   forms={forms}"))
    print()
    stale_hits = 0
    for s in stale:
        for k, t in texts.items():
            if s in t:
                stale_hits += 1
                print(f"STALE string still present in {k}: {s!r}")
    print(f"\n{len(checks)} checks, {missing} missing, {stale_hits} stale strings")
    return 1 if (missing or stale_hits) else 0


if __name__ == "__main__":
    sys.exit(main())
