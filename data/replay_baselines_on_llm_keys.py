"""Re-evaluate the four classical baselines on exactly the same chart-prefix
keys the LLMs saw, so the hybrid analysis can join across models.

Reads the LLM run, extracts the (chart_id, prefix_len) keys, regenerates
the synthetic corpus + train split with the same seed, then evaluates B0
B1 B2 B3 only on those keys. Writes a fresh results.json keyed identically.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import synth_charts  # noqa: E402
from models.base import expand_charts, patient_split  # noqa: E402


MODEL_CLASSES = {
    "b0_bigram": ("models.b0_bigram", "BigramModel"),
    "b1_tfidf_lr": ("models.b1_tfidf_lr", "TfidfLrModel"),
    "b2_xgboost": ("models.b2_xgboost", "XGBoostModel"),
    "b3_multitp": ("models.b3_multitp", "MultiTPModel"),
}


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _instantiate(slug: str):
    import importlib
    mod_name, cls_name = MODEL_CLASSES[slug]
    return getattr(importlib.import_module(mod_name), cls_name)()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm-run", type=pathlib.Path, required=True)
    p.add_argument("--n-charts", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+",
                   default=["b0_bigram", "b1_tfidf_lr", "b2_xgboost", "b3_multitp"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=pathlib.Path, default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = HERE / "cache" / f"synth_n{args.n_charts}_seed{args.seed}.json"
    if cache.exists():
        charts = synth_charts.load(cache)
    else:
        charts = synth_charts.generate_corpus(args.n_charts, seed=args.seed)

    train_charts, _, test_charts = patient_split(charts, seed=args.seed)
    train_ex = expand_charts(train_charts)
    test_ex = expand_charts(test_charts)

    # Pull the LLM keys
    llm_results = json.loads((args.llm_run / "results.json").read_text())
    keys_wanted = {(r["chart_id"], r["prefix_len"]) for r in llm_results}
    print(f"  LLM keys to replay: {len(keys_wanted)}")

    selected = [
        ex for ex in test_ex
        if (ex.chart_id, len(ex.history)) in keys_wanted
    ]
    print(f"  matched test examples: {len(selected)}")
    if not selected:
        print("ERROR: no matches — synth corpus / split must differ.")
        return 1

    out = args.out or (HERE / f"results_baseline_replay_{_utcnow()}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"  writing → {out}")

    all_trials = []
    summaries = []
    for slug in args.models:
        print(f"  → fit {slug}")
        model = _instantiate(slug)
        model.fit(train_ex)
        print(f"  → predict {slug} on {len(selected)} examples")
        correct_at = {1: 0, 3: 0, 5: 0}
        cat_total: dict[str, int] = {}
        cat_correct: dict[str, int] = {}
        from cdt_codes import category_for as _cat
        for ex in selected:
            preds = model.predict(ex, top_k=args.top_k)
            codes = [c for c, _ in preds]
            probs = [p for _, p in preds]
            for k in (1, 3, 5):
                if ex.gold in codes[:k]:
                    correct_at[k] += 1
            cat = _cat(ex.gold)
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if codes and codes[0] == ex.gold:
                cat_correct[cat] = cat_correct.get(cat, 0) + 1
            all_trials.append({
                "chart_id": ex.chart_id,
                "prefix_len": len(ex.history),
                "model": model.name,
                "gold": ex.gold,
                "gold_category": cat,
                "top_k_codes": codes,
                "top_k_probs": [round(p, 6) for p in probs],
                "correct@1": int(codes[:1] == [ex.gold]),
                "correct@3": int(ex.gold in codes[:3]),
                "correct@5": int(ex.gold in codes[:5]),
            })
        n = max(len(selected), 1)
        summaries.append({
            "model": model.name,
            "n_test_examples": len(selected),
            "top1_acc": correct_at[1] / n,
            "top3_acc": correct_at[3] / n,
            "top5_acc": correct_at[5] / n,
            "top1_by_category": {
                cat: round(cat_correct.get(cat, 0) / cat_total[cat], 4)
                for cat in cat_total
            },
            "category_counts": cat_total,
        })
        print(f"    top1={summaries[-1]['top1_acc']:.3f} top3={summaries[-1]['top3_acc']:.3f} top5={summaries[-1]['top5_acc']:.3f}")

    (out / "config.json").write_text(json.dumps(
        {"replayed_from": str(args.llm_run), "n_test_examples": len(selected),
         "args": vars(args) | {"out": str(out)}},
        indent=2, default=str,
    ))
    (out / "results.json").write_text(json.dumps(all_trials, indent=2))
    (out / "summary.json").write_text(json.dumps({"runs": summaries}, indent=2))
    print(f"\n→ {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
