"""Merge per-trial results.json from multiple runs into a single results dir.

Useful when baselines and LLM models were run in separate evaluate.py
invocations (because LLM runs cap test-set size differently). Produces a
combined ``results.json``, a unified ``summary.json`` (concatenating per-
model summaries), and a ``config.json`` listing the source runs.

Usage::

    python data/combine_results.py \
        --runs data/results_full_<bsline> data/results_full_<llm> \
        --out  data/results_combined_<ts>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, default=None)
    args = p.parse_args()

    out = args.out or pathlib.Path("data") / f"results_combined_{_utcnow()}"
    out.mkdir(parents=True, exist_ok=True)

    all_trials: list[dict] = []
    all_summaries: list[dict] = []
    sources: list[str] = []
    for run in args.runs:
        run = run.resolve()
        sources.append(str(run))
        results_path = run / "results.json"
        summary_path = run / "summary.json"
        if not results_path.exists() or not summary_path.exists():
            print(f"  (skipping {run}: missing results.json or summary.json)")
            continue
        all_trials.extend(json.loads(results_path.read_text()))
        s = json.loads(summary_path.read_text())
        all_summaries.extend(s.get("runs", s) if isinstance(s, dict) else s)

    (out / "results.json").write_text(json.dumps(all_trials, indent=2))
    (out / "summary.json").write_text(json.dumps({"runs": all_summaries}, indent=2))
    (out / "config.json").write_text(json.dumps(
        {
            "merged_from": sources,
            "merged_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "n_trials": len(all_trials),
            "n_models": len(all_summaries),
        },
        indent=2,
    ))
    print(f"→ merged {len(all_trials)} trials across {len(all_summaries)} models")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
