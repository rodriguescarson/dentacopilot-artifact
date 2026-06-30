"""Plot Paper 7 results from a results_*/ directory.

Produces:
    figures/fig1_topk_accuracy.pdf  (LaTeX-embedded)
    figures/fig1_topk_accuracy.png  (DOCX-embedded; same content)

Usage::

    python data/plot_results.py --run data/results_full_<ts>
    python data/plot_results.py --latest         # auto-pick newest results_*
    python data/plot_results.py --latest --kind smoke   # newest results_smoke_*
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
FIG_DIR = PAPER_DIR / "figures"


def _latest(prefix: str = "results_") -> pathlib.Path:
    candidates = sorted(HERE.glob(f"{prefix}*"))
    if not candidates:
        raise SystemExit(f"no {prefix}* directories found under {HERE}")
    return candidates[-1]


def _load(run_dir: pathlib.Path) -> tuple[list[dict], dict]:
    summary = json.loads((run_dir / "summary.json").read_text())["runs"]
    config = json.loads((run_dir / "config.json").read_text())
    return summary, config


def fig1_topk_accuracy(run_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    summary, config = _load(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [s["model"] for s in summary]
    top1 = [s["top1_acc"] for s in summary]
    top3 = [s["top3_acc"] for s in summary]
    top5 = [s["top5_acc"] for s in summary]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(models))
    width = 0.25
    bars1 = ax.bar(x - width, top1, width, label="Top-1", color="#3a6ea5")
    bars3 = ax.bar(x,         top3, width, label="Top-3", color="#88a0bf")
    bars5 = ax.bar(x + width, top5, width, label="Top-5", color="#c4d2e0")

    ax.set_ylabel("Accuracy")
    n_examples = (
        config.get("n_test_examples")
        or config.get("n_trials")
        or sum(s["n_test_examples"] for s in summary)
    )
    args_block = config.get("args", {})
    n_charts = args_block.get("n_charts") or "?"
    seed = args_block.get("seed") or "?"
    ax.set_title(
        f"Top-K next-procedure accuracy on {n_examples} examples\n"
        f"(synthetic chart corpus, n_charts={n_charts}, seed={seed})",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    for bars in (bars1, bars3, bars5):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_topk_accuracy.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_topk_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'fig1_topk_accuracy.pdf'}")
    print(f"  → {out_dir / 'fig1_topk_accuracy.png'}")


def fig2_top1_by_category(run_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    summary, config = _load(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = sorted({c for s in summary for c in s["top1_by_category"].keys()})
    if not cats:
        return
    models = [s["model"] for s in summary]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(cats))
    for i, s in enumerate(summary):
        vals = [s["top1_by_category"].get(c, 0.0) for c in cats]
        ax.bar(x + (i - (len(summary) - 1) / 2) * width, vals, width, label=s["model"])

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Top-1 accuracy by CDT category", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_top1_by_category.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_top1_by_category.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'fig2_top1_by_category.pdf'}")
    print(f"  → {out_dir / 'fig2_top1_by_category.png'}")


def fig3_reliability(run_dir: pathlib.Path, out_dir: pathlib.Path, n_bins: int = 10) -> None:
    """Reliability diagram: confidence vs accuracy, per model."""
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    from calibration import from_results_jsonl, reliability_curve

    results = json.loads((run_dir / "results.json").read_text())
    by_model: dict[str, list[dict]] = {}
    for ent in results:
        by_model.setdefault(ent["model"], []).append(ent)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")

    palette = ["#1b4f72", "#c0392b", "#117a65", "#7d3c98", "#b9770e", "#1f618d"]
    for i, (model_name, entries) in enumerate(sorted(by_model.items())):
        data = from_results_jsonl(entries)
        confs, accs, counts = reliability_curve(data, n_bins=n_bins)
        valid = counts > 0
        if not valid.any():
            continue
        ax.plot(
            confs[valid],
            accs[valid],
            marker="o",
            color=palette[i % len(palette)],
            label=f"{model_name} (n={data.confidences.size})",
        )

    ax.set_xlabel("Predicted top-1 confidence")
    ax.set_ylabel("Empirical accuracy on top-1")
    ax.set_title(f"Reliability diagram ({run_dir.name})", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_reliability.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'fig3_reliability.pdf'}")
    print(f"  → {out_dir / 'fig3_reliability.png'}")


def fig4_coverage_risk(run_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Coverage-risk curves: as we raise the abstention threshold,
    coverage drops and (hopefully) precision rises."""
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    from abstention import coverage_risk_curve

    results = json.loads((run_dir / "results.json").read_text())
    by_model: dict[str, list[dict]] = {}
    for ent in results:
        by_model.setdefault(ent["model"], []).append(ent)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    palette = ["#1b4f72", "#c0392b", "#117a65", "#7d3c98", "#b9770e", "#1f618d"]
    for i, (model_name, entries) in enumerate(sorted(by_model.items())):
        confs = np.array([
            (e.get("top_k_probs") or [0.0])[0] for e in entries
        ])
        corr = np.array([int(e.get("correct@1", 0)) for e in entries])
        cr = coverage_risk_curve(confs, corr)
        ax.plot(
            cr.coverage,
            cr.risk,
            color=palette[i % len(palette)],
            label=model_name,
        )

    ax.set_xlabel("Coverage (1 − abstain rate)")
    ax.set_ylabel("Risk (1 − top-1 accuracy on covered)")
    ax.set_title("Coverage–risk curve under top-1 abstention", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_coverage_risk.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_coverage_risk.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'fig4_coverage_risk.pdf'}")
    print(f"  → {out_dir / 'fig4_coverage_risk.png'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=pathlib.Path, default=None,
                   help="explicit results_* directory")
    p.add_argument("--latest", action="store_true",
                   help="auto-pick the newest results_* directory")
    p.add_argument("--kind", default="any", choices=["any", "smoke", "full"])
    p.add_argument("--out", type=pathlib.Path, default=FIG_DIR)
    p.add_argument("--no-cal", action="store_true",
                   help="skip reliability + coverage-risk figures")
    p.add_argument("--cal-run", type=pathlib.Path, default=None,
                   help="use a separate run dir for fig3+fig4 (larger n preferred)")
    args = p.parse_args()

    if args.run is None:
        prefix = {"any": "results_", "smoke": "results_smoke_", "full": "results_full_"}[args.kind]
        args.run = _latest(prefix)

    print(f"→ Using run: {args.run}")
    fig1_topk_accuracy(args.run, args.out)
    fig2_top1_by_category(args.run, args.out)
    if not args.no_cal:
        cal_run = args.cal_run or args.run
        try:
            fig3_reliability(cal_run, args.out)
            fig4_coverage_risk(cal_run, args.out)
        except Exception as exc:
            print(f"  (calibration figures skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
