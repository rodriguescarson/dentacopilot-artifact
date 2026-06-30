"""One-shot Stage 3 finalizer: run after the LLM eval completes.

1. Identify the latest LLM full-run dir (has llm_calls.jsonl).
2. Identify the latest baseline full-run dir (has b0/b1/b2 entries).
3. Compute verbalized-confidence calibration for each LLM model.
4. Merge LLM + baseline results into a combined results dir.
5. Regenerate fig1–fig4.
6. Print a summary table for the paper.

Run::

    python data/finalize_stage3.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def latest_with_llm() -> pathlib.Path | None:
    candidates = sorted(HERE.glob("results_full_*"))
    for c in reversed(candidates):
        if (c / "llm_calls.jsonl").exists() and (c / "summary.json").exists():
            return c
    return None


def latest_baseline() -> pathlib.Path | None:
    candidates = sorted(HERE.glob("results_full_*"))
    for c in reversed(candidates):
        if not (c / "summary.json").exists():
            continue
        s = json.loads((c / "summary.json").read_text())
        runs = s.get("runs", []) if isinstance(s, dict) else s
        if any(r["model"].startswith("b0") for r in runs):
            return c
    return None


def main() -> int:
    llm_run = latest_with_llm()
    bsl_run = latest_baseline()
    print(f"LLM run     : {llm_run}")
    print(f"Baseline run: {bsl_run}")
    if llm_run is None:
        print("No completed LLM run yet.")
        return 1

    # 1. verbalized confidence
    print("\n--- analyze llm_calls.jsonl ---")
    subprocess.run(
        [sys.executable, str(HERE / "analyze_llm_calls.py"), "--run", str(llm_run)],
        check=False,
    )

    # 2. merge with baseline
    if bsl_run is not None:
        from combine_results import _utcnow  # noqa: E402
        out_dir = HERE / f"results_combined_{_utcnow()}"
        print(f"\n--- merge → {out_dir.name} ---")
        subprocess.run(
            [
                sys.executable,
                str(HERE / "combine_results.py"),
                "--runs",
                str(bsl_run),
                str(llm_run),
                "--out",
                str(out_dir),
            ],
            check=False,
        )
    else:
        out_dir = llm_run

    # 3. plots
    print(f"\n--- plot_results.py --run {out_dir.name} ---")
    subprocess.run(
        [sys.executable, str(HERE / "plot_results.py"), "--run", str(out_dir)],
        check=False,
    )

    # 4. final summary table for paper
    s = json.loads((out_dir / "summary.json").read_text())
    runs = s.get("runs", []) if isinstance(s, dict) else s
    print("\n=== SUMMARY (paste into paper) ===")
    print(f"{'model':<22} {'n':>5} {'top1':>6} {'top3':>6} {'top5':>6}")
    for r in runs:
        print(
            f"{r['model']:<22} "
            f"{r['n_test_examples']:>5} "
            f"{r['top1_acc']:>6.3f} "
            f"{r['top3_acc']:>6.3f} "
            f"{r['top5_acc']:>6.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
