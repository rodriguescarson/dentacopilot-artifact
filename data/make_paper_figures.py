"""Redraw the two figures the manuscript prints (added 2026-10-10).

plot_results.py draws exploratory figures for any results_* directory and puts the run directory
and placeholder metadata in the title ("n_charts=?, seed=?", "results_full_<ts>"). The paper needs
the same data with publication labels, so this script redraws exactly those two figures from the
same sources:

  figures/fig1_topk_accuracy.{pdf,png}  <- results_matched_n30_v2/summary.json (the n=30 table)
  figures/fig3_reliability.{pdf,png}    <- results_full_2026-05-02T12-35-51Z/results.json (n=1,284)
  figures/fig2_top1_by_category.{pdf,png} <- results_matched_n30_v2/summary.json (supplement Fig. S1)
  figures/fig4_coverage_risk.{pdf,png}  <- results_full_2026-05-02T12-35-51Z/results.json (supplement Fig. S2)

No title is drawn (the caption carries it); model names use the paper's labels.

Run:  python3 data/make_paper_figures.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
sys.path.insert(0, str(HERE))
from calibration import from_results_jsonl, reliability_curve  # noqa: E402

LABELS = {"b0_bigram": "B0\nbigram", "b1_tfidf_lr": "B1\nTF-IDF\n+LR", "b2_xgboost": "B2\nXGBoost",
          "b3_multitp": "B3\nMultiTP", "m1_haiku": "M1\nHaiku", "m2_sonnet_cot": "M2\nSonnet\n+CoT",
          "m3_sonnet_rag": "M3\nSonnet\n+RAG", "m5_sonnet_baseline_primed": "M5\nSonnet\n+prior"}
ORDER = ["b0_bigram", "b1_tfidf_lr", "b2_xgboost", "b3_multitp",
         "m1_haiku", "m2_sonnet_cot", "m3_sonnet_rag", "m5_sonnet_baseline_primed"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def fig1() -> None:
    runs = {s["model"]: s for s in json.loads((HERE / "results_matched_n30_v2" / "summary.json").read_text())["runs"]}
    assert all(runs[m]["n_test_examples"] == 30 for m in ORDER)
    x = np.arange(len(ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for off, key, lab, col in ((-w, "top1_acc", "Top-1", "#2b5d8a"), (0, "top3_acc", "Top-3", "#7f9fc4"),
                               (w, "top5_acc", "Top-5", "#c9d6e6")):
        vals = [runs[m][key] for m in ORDER]
        bars = ax.bar(x + off, vals, w, label=lab, color=col, edgecolor="white", linewidth=0.4)
        if key == "top1_acc":
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7)
    ax.axvline(3.5, color="#999999", linewidth=0.8, linestyle="--")
    ax.text(1.5, 1.04, "classical baselines", ha="center", fontsize=8, color="#444444")
    ax.text(5.5, 1.04, "LLM variants", ha="center", fontsize=8, color="#444444")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("Accuracy ($n{=}30$)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_topk_accuracy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3() -> None:
    results = json.loads((HERE / "results_full_2026-05-02T12-35-51Z" / "results.json").read_text())
    by_model: dict[str, list[dict]] = {}
    for ent in results:
        by_model.setdefault(ent["model"], []).append(ent)
    fig, ax = plt.subplots(figsize=(3.7, 3.9))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, linewidth=0.9, label="Perfect calibration")
    style = {"b0_bigram": ("B0 bigram", "#1b4f72", "o"), "b1_tfidf_lr": ("B1 TF-IDF+LR", "#c0392b", "s"),
             "b2_xgboost": ("B2 XGBoost", "#117a65", "^")}
    for m, (lab, col, mk) in style.items():
        data = from_results_jsonl(by_model[m])
        conf, acc, cnt = reliability_curve(data, n_bins=10)
        v = cnt > 0
        ax.plot(conf[v], acc[v], marker=mk, markersize=4, linewidth=1.2, color=col,
                label=f"{lab} ($n{{=}}{data.confidences.size:,}$)".replace(",", "{,}"))
    ax.set_xlabel("Predicted top-1 confidence"); ax.set_ylabel("Observed top-1 accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig3_reliability.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


FLAT = {k: v.replace("\n+", "+").replace("\n", " ") for k, v in LABELS.items()}
COLORS = {"b0_bigram": "#1b4f72", "b1_tfidf_lr": "#c0392b", "b2_xgboost": "#117a65", "b3_multitp": "#7d3c98",
          "m1_haiku": "#b9770e", "m2_sonnet_cot": "#5d6d7e", "m3_sonnet_rag": "#d98880",
          "m5_sonnet_baseline_primed": "#48c9b0"}


def fig2_by_category() -> None:
    runs = {s["model"]: s for s in json.loads((HERE / "results_matched_n30_v2" / "summary.json").read_text())["runs"]}
    cats = sorted({c for s in runs.values() for c in s["top1_by_category"]})
    counts = runs["b0_bigram"]["category_counts"]
    x = np.arange(len(cats)); w = 0.8 / len(ORDER)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, m in enumerate(ORDER):
        ax.bar(x + (i - (len(ORDER) - 1) / 2) * w, [runs[m]["top1_by_category"].get(c, 0.0) for c in cats], w,
               label=FLAT[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c.replace('_', ' ')}\n($n{{=}}{counts.get(c, 0)}$)" for c in cats], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Top-1 accuracy ($n{=}30$)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_top1_by_category.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_coverage_risk() -> None:
    from abstention import coverage_risk_curve
    results = json.loads((HERE / "results_full_2026-05-02T12-35-51Z" / "results.json").read_text())
    by_model: dict[str, list[dict]] = {}
    for ent in results:
        by_model.setdefault(ent["model"], []).append(ent)
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    for m in ("b0_bigram", "b1_tfidf_lr", "b2_xgboost"):
        conf = np.array([(e.get("top_k_probs") or [0.0])[0] for e in by_model[m]])
        corr = np.array([int(e.get("correct@1", 0)) for e in by_model[m]])
        cr = coverage_risk_curve(conf, corr)
        ax.plot(cr.coverage, cr.risk, color=COLORS[m], linewidth=1.3, label=FLAT[m])
    ax.set_xlabel("Coverage (1 $-$ abstention rate)"); ax.set_ylabel("Risk on covered cases (1 $-$ top-1)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig4_coverage_risk.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig3(); fig2_by_category(); fig4_coverage_risk()
    print("wrote figures/fig1_topk_accuracy, fig3_reliability, fig2_top1_by_category, fig4_coverage_risk (.pdf/.png)")
