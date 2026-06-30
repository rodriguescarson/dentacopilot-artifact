"""Analyze llm_calls.jsonl: pair each call with its meta entry, build per-trial
records, compute verbalized-confidence calibration and abstention statistics.

Usage::

    python data/analyze_llm_calls.py --run data/results_full_<ts>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calibration import calibrate_verbalized, map_verbalized_confidence  # noqa: E402


def pair_calls_and_meta(jsonl_path: pathlib.Path) -> list[dict]:
    """Pair each prediction (call) with its meta entry by run_id."""
    by_run: dict[str, dict] = defaultdict(dict)
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        run_id = entry.get("run_id")
        if run_id is None:
            continue
        if "meta" in entry:
            by_run[run_id]["meta"] = entry["meta"]
        else:
            for key in ("ts", "model", "prompt", "response", "latency_ms",
                        "tokens_in", "tokens_out"):
                if key in entry:
                    by_run[run_id][key] = entry[key]
    return [{"run_id": k, **v} for k, v in by_run.items()]


def attach_to_results(per_trial: list[dict], paired: list[dict]) -> list[dict]:
    """Add the verbalized confidence + abstain flag from the paired LLM
    metadata to each trial-results entry. Returns a new list of dicts."""
    by_run: dict[str, dict] = {p["run_id"]: p for p in paired}
    enriched: list[dict] = []
    for ent in per_trial:
        ent2 = dict(ent)
        # The trial doesn't itself record run_id; reconstruct it
        # (matches the convention in llm_base.predict)
        # We can't reconstruct without call_index. Skip this attach when not LLM.
        enriched.append(ent2)
    return enriched


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=pathlib.Path, required=True)
    args = p.parse_args()

    jsonl = args.run / "llm_calls.jsonl"
    if not jsonl.exists():
        print(f"No llm_calls.jsonl at {jsonl} — nothing to analyze.")
        return 1

    paired = pair_calls_and_meta(jsonl)
    print(f"→ Paired {len(paired)} (call, meta) records.")

    # Group by model and compute verbalized-confidence calibration
    by_model: dict[str, list[dict]] = defaultdict(list)
    results = json.loads((args.run / "results.json").read_text())
    by_model_trial: dict[str, list[dict]] = defaultdict(list)
    for ent in results:
        by_model_trial[ent["model"]].append(ent)
    for rec in paired:
        # Run id pattern: "{model_name}-{chart_id}-prefix{N}-call{K}"
        rid = rec["run_id"]
        head = rid.split("-prefix")[0]
        model_name = head.rsplit("-", 1)[0] if "-" in head else head
        by_model[model_name].append(rec)

    for model_name, recs in sorted(by_model.items()):
        print(f"\n[{model_name}]  n_calls={len(recs)}")
        # Pair the i-th rec with i-th trial (both produced in iteration order)
        trials = by_model_trial.get(model_name, [])
        if len(trials) != len(recs):
            print(
                f"  warn: trial count {len(trials)} != call count {len(recs)} — "
                f"skipping per-confidence stats"
            )
            continue
        pairs: list[tuple[str | None, int]] = []
        n_abstain = 0
        n_parse_fail = 0
        for rec, trial in zip(recs, trials):
            meta = rec.get("meta", {})
            conf = meta.get("confidence")
            if meta.get("abstain"):
                n_abstain += 1
            if meta.get("parse_failed"):
                n_parse_fail += 1
            pairs.append((conf, int(trial.get("correct@1", 0))))
        bucket = calibrate_verbalized(pairs)
        print(f"  abstain rate    : {n_abstain}/{len(recs)} ({n_abstain/max(len(recs),1):.2%})")
        print(f"  parse fail rate : {n_parse_fail}/{len(recs)} ({n_parse_fail/max(len(recs),1):.2%})")
        for label in ("low", "medium", "high"):
            b = bucket.get(label, {})
            n = b.get("n", 0)
            acc = b.get("acc", float("nan"))
            print(f"  {label:6s} verbalized: n={n:4d}  empirical_top1={acc:.3f}")

    out = args.run / "verbalized_calibration.json"
    summary = {
        model_name: {
            "n_calls": len(recs),
            "n_abstain": sum(1 for r in recs if r.get("meta", {}).get("abstain")),
            "n_parse_fail": sum(1 for r in recs if r.get("meta", {}).get("parse_failed")),
        }
        for model_name, recs in by_model.items()
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
