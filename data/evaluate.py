"""Evaluation harness for Paper 7 (DentaCoPilot) — single CLI entry point.

Trains and evaluates one or more models on a chart-history dataset, writing
a fully-reproducible run record:

    data/results_<descriptor>/
        config.json     git SHA, seed, CLI args, model versions
        results.json    per-trial: {chart_id, prefix_len, model, gold,
                                    top_k_codes, top_k_probs, correct@1,
                                    correct@3, correct@5}
        summary.json    aggregated metrics per model
        llm_calls.jsonl appended by cc_compute.py for LLM-backed models

Usage::

    python data/evaluate.py --models b0 b1 b2 --n-charts 200 --smoke
    python data/evaluate.py --models b0 b1 b2 --n-charts 1000

The harness deliberately avoids any LLM call in baseline-only mode so
``--smoke`` runs in seconds without burning Claude Code quota.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import synth_charts  # noqa: E402
from models.base import Example, Model, expand_charts, patient_split  # noqa: E402

MODEL_REGISTRY = {
    "b0": ("models.b0_bigram", "BigramModel"),
    "b1": ("models.b1_tfidf_lr", "TfidfLrModel"),
    "b2": ("models.b2_xgboost", "XGBoostModel"),
    "b3": ("models.b3_multitp", "MultiTPModel"),
    "m1": ("models.m1_haiku", "HaikuModel"),
    "m2": ("models.m2_sonnet_cot", "SonnetCoTModel"),
    "m3": ("models.m3_sonnet_rag", "SonnetRagModel"),
    "m4": ("models.m4_opus", "OpusCoTModel"),
    "m5": ("models.m5_sonnet_baseline_primed", "SonnetBaselinePrimedModel"),
    "m6": ("models.m6_opus_baseline_primed", "OpusBaselinePrimedModel"),
}

LLM_MODELS = {"m1", "m2", "m3", "m4", "m5", "m6"}


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent.parent, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _instantiate(slug: str) -> Model:
    if slug not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {slug!r}; valid: {list(MODEL_REGISTRY)}")
    module_name, class_name = MODEL_REGISTRY[slug]
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


def _utcnow_compact() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _evaluate_one(
    model: Model,
    test_examples: Sequence[Example],
    top_k: int,
) -> tuple[list[dict], dict]:
    per_trial: list[dict] = []
    correct_at = {1: 0, 3: 0, 5: 0}
    cat_total: dict[str, int] = {}
    cat_correct: dict[str, int] = {}
    started = time.perf_counter()

    from cdt_codes import category_for as _cat

    for ex in test_examples:
        preds = model.predict(ex, top_k=top_k)
        codes = [c for c, _ in preds]
        probs = [p for _, p in preds]
        for k in (1, 3, 5):
            if ex.gold in codes[:k]:
                correct_at[k] += 1
        cat = _cat(ex.gold)
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if codes and codes[0] == ex.gold:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
        per_trial.append(
            {
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
            }
        )

    n = max(len(test_examples), 1)
    elapsed = time.perf_counter() - started
    summary = {
        "model": model.name,
        "n_test_examples": len(test_examples),
        "top1_acc": correct_at[1] / n,
        "top3_acc": correct_at[3] / n,
        "top5_acc": correct_at[5] / n,
        "top1_by_category": {
            cat: round(cat_correct.get(cat, 0) / cat_total[cat], 4)
            for cat in cat_total
        },
        "category_counts": cat_total,
        "wallclock_s": round(elapsed, 3),
    }
    return per_trial, summary


def _load_or_build_corpus(args: argparse.Namespace) -> list[synth_charts.Chart]:
    cache = HERE / "cache" / f"synth_n{args.n_charts}_seed{args.seed}.json"
    if cache.exists() and not args.regenerate:
        return synth_charts.load(cache)
    charts = synth_charts.generate_corpus(args.n_charts, seed=args.seed)
    synth_charts.save(charts, cache)
    return charts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["b0", "b1", "b2"])
    p.add_argument("--data-source", default="synthetic", choices=["synthetic"])
    p.add_argument("--n-charts", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 100 charts, ≤500 test examples")
    p.add_argument("--regenerate", action="store_true",
                   help="force re-generation of the synthetic corpus")
    p.add_argument("--results-root", type=pathlib.Path, default=HERE / "results")
    p.add_argument("--max-test", type=int, default=2000,
                   help="cap on test examples (random subsample) to keep runtime bounded")
    p.add_argument("--llm-cap", type=int, default=200,
                   help="separate cap for LLM models (each call spends Claude Code quota)")
    args = p.parse_args()

    if args.smoke:
        args.n_charts = min(args.n_charts, 100)
        args.max_test = min(args.max_test, 500)
        args.llm_cap = min(args.llm_cap, 10)

    random.seed(args.seed)
    np.random.seed(args.seed)

    descriptor = "smoke" if args.smoke else "full"
    run_dir = args.results_root.parent / f"results_{descriptor}_{_utcnow_compact()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Run dir: {run_dir}")

    charts = _load_or_build_corpus(args)
    train_charts, dev_charts, test_charts = patient_split(charts, seed=args.seed)
    train_ex = expand_charts(train_charts)
    dev_ex = expand_charts(dev_charts)
    test_ex = expand_charts(test_charts)

    if args.max_test and len(test_ex) > args.max_test:
        rng = random.Random(args.seed)
        test_ex = rng.sample(test_ex, args.max_test)

    print(f"→ Charts: {len(train_charts)} train / {len(dev_charts)} dev / {len(test_charts)} test")
    print(f"→ Examples: {len(train_ex)} train / {len(test_ex)} test")

    config = {
        "git_sha": _git_sha(),
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "args": vars(args) | {"results_root": str(args.results_root)},
        "n_train_examples": len(train_ex),
        "n_test_examples": len(test_ex),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # LLM models share an evaluation subsample so accuracy is apples-to-apples
    llm_test_ex = test_ex
    if args.llm_cap and len(test_ex) > args.llm_cap:
        rng = random.Random(args.seed + 1)
        llm_test_ex = rng.sample(test_ex, args.llm_cap)

    all_trials: list[dict] = []
    summaries: list[dict] = []
    for slug in args.models:
        print(f"→ {slug} :: fit")
        model = _instantiate(slug)
        # LLM models log every call to the run dir; attach the logger.
        if slug in LLM_MODELS and hasattr(model, "attach_logger"):
            model.attach_logger(run_dir)
        t0 = time.perf_counter()
        model.fit(train_ex)
        fit_s = time.perf_counter() - t0
        print(f"   fit took {fit_s:.2f}s")

        eval_set = llm_test_ex if slug in LLM_MODELS else test_ex
        print(f"→ {slug} :: evaluate (n={len(eval_set)})")
        trials, summary = _evaluate_one(model, eval_set, top_k=args.top_k)
        summary["fit_wallclock_s"] = round(fit_s, 3)
        summary["used_llm_cap"] = slug in LLM_MODELS
        summaries.append(summary)
        all_trials.extend(trials)
        print(
            f"   top1={summary['top1_acc']:.3f}  "
            f"top3={summary['top3_acc']:.3f}  "
            f"top5={summary['top5_acc']:.3f}"
        )

    (run_dir / "results.json").write_text(json.dumps(all_trials, indent=2))
    (run_dir / "summary.json").write_text(json.dumps({"runs": summaries}, indent=2))

    print(f"\nDone. {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
