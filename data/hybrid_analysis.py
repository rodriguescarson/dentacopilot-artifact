"""Confidence-routed hybrid analysis: baseline when confident, LLM when not.

Hypothesis: classical baselines win on bundled / sequential procedures
because the local Markov signal is very strong; LLMs win where chart
context matters (perio, preventive). A simple gating rule — defer to the
LLM only when the baseline's top-1 confidence is below a threshold ---
should beat both pure approaches.

We compute this purely from existing per-trial JSON; no new LLM calls.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def trials_by_model(results_path: pathlib.Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for ent in json.loads(results_path.read_text()):
        out[ent["model"]].append(ent)
    return out


def by_key(trials: list[dict]) -> dict[tuple, dict]:
    """Index trials by (chart_id, prefix_len) for joining across models."""
    return {(t["chart_id"], t["prefix_len"]): t for t in trials}


def hybrid_topk(
    base_trial: dict, llm_trial: dict, threshold: float, top_k: int = 5
) -> tuple[list[str], str]:
    base_top1_prob = (base_trial.get("top_k_probs") or [0.0])[0]
    if base_top1_prob >= threshold:
        codes = base_trial["top_k_codes"][:top_k]
        return codes, "base"
    codes = llm_trial["top_k_codes"][:top_k]
    return codes, "llm"


def evaluate_hybrid(
    base_trials: list[dict],
    llm_trials: list[dict],
    threshold: float,
    top_k: int = 5,
) -> dict:
    base_idx = by_key(base_trials)
    llm_idx = by_key(llm_trials)
    keys = sorted(set(base_idx) & set(llm_idx))
    if not keys:
        return {"n": 0, "error": "no overlap"}

    base_used = 0
    correct_at = {1: 0, 3: 0, 5: 0}
    base_corr = sum(t["correct@1"] for k in keys for t in [base_idx[k]])
    llm_corr  = sum(t["correct@1"] for k in keys for t in [llm_idx[k]])

    for key in keys:
        b = base_idx[key]
        l = llm_idx[key]
        codes, source = hybrid_topk(b, l, threshold, top_k=top_k)
        if source == "base":
            base_used += 1
        gold = b["gold"]
        if gold == codes[0]:
            correct_at[1] += 1
        if gold in codes[:3]:
            correct_at[3] += 1
        if gold in codes[:5]:
            correct_at[5] += 1

    n = len(keys)
    return {
        "n": n,
        "base_share": round(base_used / n, 3),
        "llm_share": round((n - base_used) / n, 3),
        "hybrid_top1": round(correct_at[1] / n, 3),
        "hybrid_top3": round(correct_at[3] / n, 3),
        "hybrid_top5": round(correct_at[5] / n, 3),
        "pure_base_top1": round(base_corr / n, 3),
        "pure_llm_top1": round(llm_corr / n, 3),
        "threshold": threshold,
    }


def sweep(base_trials: list[dict], llm_trials: list[dict]) -> list[dict]:
    rows = []
    for tau in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
        rows.append(evaluate_hybrid(base_trials, llm_trials, threshold=tau))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results", type=pathlib.Path,
        default=pathlib.Path("data/results_combined_n30_all/results.json"),
    )
    p.add_argument("--base", default="b0_bigram")
    p.add_argument("--llm", default="m2_sonnet_cot")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    by_model = trials_by_model(args.results)
    base_trials = by_model.get(args.base, [])
    llm_trials = by_model.get(args.llm, [])
    if not base_trials or not llm_trials:
        print(f"Need both {args.base} and {args.llm} in {args.results}")
        return 1

    print(f"Hybrid: {args.base}  →  {args.llm} (when base prob < τ)")
    print(f"{'τ':>5} {'n':>4} {'base%':>6} {'llm%':>6} {'hybrid_top1':>12} "
          f"{'hybrid_top3':>12} {'hybrid_top5':>12}")
    rows = sweep(base_trials, llm_trials)
    for row in rows:
        print(f"{row['threshold']:>5.2f} {row['n']:>4} "
              f"{row['base_share']:>6.2f} {row['llm_share']:>6.2f} "
              f"{row['hybrid_top1']:>12.3f} {row['hybrid_top3']:>12.3f} "
              f"{row['hybrid_top5']:>12.3f}")
    print(f"\nPure base ({args.base})  top1: {rows[0]['pure_base_top1']:.3f}")
    print(f"Pure LLM  ({args.llm})    top1: {rows[0]['pure_llm_top1']:.3f}")

    # Save best-threshold result
    best = max(rows, key=lambda r: r.get("hybrid_top1", 0))
    out = args.results.parent / f"hybrid_{args.base}__{args.llm}.json"
    out.write_text(json.dumps({
        "base_model": args.base,
        "llm_model": args.llm,
        "sweep": rows,
        "best": best,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
