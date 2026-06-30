"""Plot the hybrid confidence-routing trade-off.

X axis: threshold τ (fraction routed to LLM rises with τ).
Y axis: top-K accuracy.

One panel per (base, llm) pair.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matched", type=pathlib.Path,
                   default=HERE / "results_matched_n30")
    p.add_argument("--out", type=pathlib.Path,
                   default=HERE.parent / "figures")
    args = p.parse_args()

    pairs = [
        ("b0_bigram", "m2_sonnet_cot"),
        ("b1_tfidf_lr", "m2_sonnet_cot"),
        ("b3_multitp", "m2_sonnet_cot"),
        ("b0_bigram", "m3_sonnet_rag"),
    ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(16, 4), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (base, llm) in zip(axes, pairs):
        path = args.matched / f"hybrid_{base}__{llm}.json"
        if not path.exists():
            ax.set_title(f"missing\n{path.name}")
            continue
        data = json.loads(path.read_text())
        sweep = data["sweep"]
        taus = [s["threshold"] for s in sweep]
        h1 = [s["hybrid_top1"] for s in sweep]
        h3 = [s["hybrid_top3"] for s in sweep]
        h5 = [s["hybrid_top5"] for s in sweep]
        ax.plot(taus, h1, "-o", label="hybrid top-1", color="#1b4f72")
        ax.plot(taus, h3, "-o", label="hybrid top-3", color="#3498db")
        ax.plot(taus, h5, "-o", label="hybrid top-5", color="#aed6f1")
        # Reference lines
        ax.axhline(sweep[0]["pure_base_top1"], color="#138d75", linestyle="--",
                   label=f"pure base top-1", alpha=0.7)
        ax.axhline(sweep[0]["pure_llm_top1"], color="#c0392b", linestyle="--",
                   label=f"pure LLM top-1", alpha=0.7)
        ax.set_xlabel("τ (route to LLM when base prob < τ)")
        ax.set_title(f"{base} → {llm}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Confidence-routed hybrid: classical baseline (gate) → LLM "
        "(when base top-1 < τ).  Synthetic n=30.",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig5_hybrid.pdf", bbox_inches="tight")
    fig.savefig(args.out / "fig5_hybrid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {args.out / 'fig5_hybrid.pdf'}")
    print(f"  → {args.out / 'fig5_hybrid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
