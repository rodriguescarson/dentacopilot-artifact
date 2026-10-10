"""Jev Router arm (added 2026-10-10): the two existing LLM probes, re-run with
TypeSafe's Jev Router through OpenRouter, paired example-for-example with the
Claude runs already reported.

Arm ``synth``: the category-stratified n=200 sample of ``rerun_stratified_n200``
(same corpus, split seed 42, sample seed 2026), with the two prompts that run
used: M2 (chain-of-thought) and M5 (conditioned on the TF-IDF + LR model's
top-10 candidates). Prompts are rebuilt with the same model classes and
checked byte-for-byte against the prompts logged in
``results_stratified_n200/llm_calls.jsonl`` before any call is made.

Arm ``meps``: the 150-transition MEPS 2023 probe of ``meps_llm_eval`` (seed 42,
held-out 30% split, the same thin-feature prompt the Haiku probe used).

Every call is logged to ``<out>/llm_calls.jsonl`` with the routed model,
provider and billed cost (see ``openrouter_compute``). The run stops itself if
the cumulative cost passes ``--cost-cap``.

Run:
  OPENROUTER_ENV_FILE=~/Projects/.env python3 run_jev_openrouter.py --arm synth
  OPENROUTER_ENV_FILE=~/Projects/.env python3 run_jev_openrouter.py --arm meps
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import platform
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from openrouter_compute import JEV_MODEL, JsonlLog, _api_key, or_call  # noqa: E402

RETRY_SLEEPS = (10, 60, 180)
JEV_DESIGNS = {  # Jev design name -> the Sonnet design whose prompt it reuses
    "m7_jev_cot": "m2_sonnet_cot",
    "m8_jev_primed": "m5_sonnet_baseline_primed",
}


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Runner:
    def __init__(self, out: pathlib.Path, workers: int, cost_cap: float) -> None:
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.log_calls = JsonlLog(out / "llm_calls.jsonl")
        self.partial = out / "trials_partial.jsonl"
        self.workers, self.cost_cap = workers, cost_cap
        self.lock = threading.Lock()
        self.cost = 0.0
        self.abort = False
        if (out / "llm_calls.jsonl").exists():
            for line in (out / "llm_calls.jsonl").read_text().splitlines():
                self.cost += json.loads(line).get("cost_usd") or 0.0

    def progress(self, msg: str) -> None:
        line = f"{_stamp()} {msg}"
        print(line, flush=True)
        with (self.out / "progress.log").open("a") as f:
            f.write(line + "\n")

    def call(self, prompt: str, run_id: str):
        last = None
        for attempt, sleep_s in enumerate((0,) + RETRY_SLEEPS):
            if self.abort:
                raise RuntimeError("aborted: cost cap")
            if sleep_s:
                time.sleep(sleep_s)
            try:
                res = or_call(prompt, run_id=f"{run_id}-try{attempt}", log=self.log_calls)
            except RuntimeError as exc:
                last = str(exc)[:300]
                continue
            with self.lock:
                self.cost += res.cost_usd or 0.0
                if self.cost > self.cost_cap:
                    self.abort = True
            return res, attempt
        raise RuntimeError(last or "unknown")

    def done_keys(self) -> tuple[list[dict], set]:
        rows = []
        if self.partial.exists():
            rows = [json.loads(l) for l in self.partial.read_text().splitlines()]
        return rows, {r["key"] for r in rows}

    def run_tasks(self, tasks, fn) -> list[dict]:
        rows, done = self.done_keys()
        todo = [t for t in tasks if t["key"] not in done]
        self.progress(f"START {len(todo)} calls ({len(done)} already done) cost_so_far=${self.cost:.4f}")
        t0 = time.perf_counter()
        n = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = [pool.submit(fn, t) for t in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                with self.lock:
                    rows.append(rec)
                    with self.partial.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    n += 1
                el = time.perf_counter() - t0
                self.progress(f"{n}/{len(todo)} {rec['key']} routed={rec.get('routed_model')} "
                              f"ok={'error' not in rec} cost=${self.cost:.4f} "
                              f"eta={(el / n * (len(todo) - n)) / 60:.1f}min")
                if self.abort:
                    self.progress(f"ABORT: cost cap ${self.cost_cap} passed")
                    for f2 in futs:
                        f2.cancel()
                    break
        return rows


def arm_synth(args) -> int:
    import rerun_stratified_n200 as rs
    import synth_charts
    from models.base import expand_charts, patient_split
    from models.llm_base import _parse_predictions

    random.seed(42); np.random.seed(42)
    charts = synth_charts.load(HERE / "cache" / "synth_n500_seed42.json")
    tr, _dev, te = patient_split(charts, seed=42)
    train_ex, test_ex = expand_charts(tr), expand_charts(te)
    chosen, plan = rs.stratified_sample(test_ex, random.Random(rs.SAMPLE_SEED))
    prompt_models = {}
    for slug in set(JEV_DESIGNS.values()):
        m = rs._instantiate(rs.LLMS[slug]); m.fit(train_ex); prompt_models[slug] = m

    # Byte-for-byte prompt check against the logged Sonnet run.
    logged = {}
    for line in (HERE / "results_stratified_n200" / "llm_calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "prompt" in r:
            logged.setdefault(r["run_id"].rsplit("-try", 1)[0], r["prompt"])
    tasks, match, missing = [], 0, 0
    for idx, ex in enumerate(chosen):
        for jev_name, slug in JEV_DESIGNS.items():
            m = prompt_models[slug]
            prompt = m._build_prompt(ex, 5)
            ref = logged.get(f"{m.name}-{ex.chart_id}-prefix{len(ex.history)}-strat{idx}")
            if ref is None:
                missing += 1
            elif ref == prompt:
                match += 1
            else:
                raise SystemExit(f"prompt mismatch for {m.name} {ex.chart_id} strat{idx}")
            tasks.append({"key": f"{jev_name}|{ex.chart_id}|{len(ex.history)}", "design": jev_name,
                          "ex": ex, "idx": idx, "prompt": prompt, "allowed": m._allowed_codes})
    runner = Runner(args.out, args.workers, args.cost_cap)
    cfg = {"started_utc": _stamp(), "arm": "synth", "requested_model": JEV_MODEL,
           "designs": JEV_DESIGNS, "plan": plan, "n_chosen": len(chosen),
           "prompt_check": {"identical_to_logged_sonnet_prompt": match, "not_in_log": missing},
           "cost_cap_usd": args.cost_cap, "workers": args.workers,
           "python": platform.python_version()}
    if not (args.out / "config.json").exists():
        (args.out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    runner.progress(f"prompt check: {match} identical, {missing} not in log, 0 mismatched")

    def one(t):
        ex = t["ex"]
        run_id = f"{t['design']}-{ex.chart_id}-prefix{len(ex.history)}-strat{t['idx']}"
        try:
            res, attempt = runner.call(t["prompt"], run_id)
        except RuntimeError as exc:
            rec = rs.trial_record(t["design"], ex, [], error=str(exc)[:300])
            rec["key"] = t["key"]
            return rec
        preds, meta = _parse_predictions(res.text, t["allowed"], 5)
        rec = rs.trial_record(t["design"], ex, preds, meta=meta)
        rec.update({"key": t["key"], "strat_idx": t["idx"], "routed_model": res.routed_model,
                    "provider": res.provider, "cost_usd": res.cost_usd,
                    "latency_ms": res.latency_ms, "attempts": attempt + 1,
                    "finish_reason": res.finish_reason})
        return rec

    rows = runner.run_tasks(tasks, one)
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    summ = rs.summarise(rows)
    (args.out / "summary.json").write_text(json.dumps(
        {"runs": summ, "aborted": runner.abort, "cost_usd": round(runner.cost, 4)}, indent=2))
    for s in summ:
        runner.progress(f"SUMMARY {s['model']} n={s['n_test_examples']} failed={s['n_failed_calls']} "
                        f"top1={s['top1_acc']:.3f} top3={s['top3_acc']:.3f} top5={s['top5_acc']:.3f}")
    runner.progress(f"DONE cost=${runner.cost:.4f}" + (" (ABORTED)" if runner.abort else ""))
    return 2 if runner.abort else 0


def arm_meps(args) -> int:
    import meps_llm_eval as me
    rng = random.Random(me.SEED)
    tr = me.build_transitions()
    idx = list(range(len(tr))); rng.shuffle(idx); cut = int(0.7 * len(idx))
    train, test = tr.iloc[idx[:cut]], tr.iloc[idx[cut:]]
    cond = train.groupby("prev_cat")["next_cat"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    glob = train["next_cat"].value_counts().idxmax()
    sample = test.sample(n=min(args.n, len(test)), random_state=me.SEED)
    runner = Runner(args.out, args.workers, args.cost_cap)
    cfg = {"started_utc": _stamp(), "arm": "meps", "requested_model": JEV_MODEL, "n": len(sample),
           "seed": me.SEED, "n_transitions": len(tr), "cost_cap_usd": args.cost_cap}
    if not (args.out / "config.json").exists():
        (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    tasks = []
    for row_idx, r in sample.iterrows():
        tasks.append({"key": f"meps|{row_idx}", "row": int(row_idx), "prev_cat": r["prev_cat"],
                      "prev_prov": r["prev_prov"], "gold": r["next_cat"],
                      "markov": cond.get(r["prev_cat"], glob),
                      "prompt": me.prompt_for(r["prev_cat"], r["prev_prov"])})

    def one(t):
        base = {k: t[k] for k in ("key", "row", "prev_cat", "prev_prov", "gold", "markov")}
        base["markov_correct"] = int(t["markov"] == t["gold"])
        try:
            res, attempt = runner.call(t["prompt"], f"jev-meps-{t['row']}")
        except RuntimeError as exc:
            return {**base, "error": str(exc)[:300]}
        pred = me.parse(res.text)
        return {**base, "pred": pred, "jev_correct": int(pred == t["gold"]), "raw": res.text[:200],
                "routed_model": res.routed_model, "provider": res.provider,
                "cost_usd": res.cost_usd, "latency_ms": res.latency_ms, "attempts": attempt + 1}

    rows = runner.run_tasks(tasks, one)
    ok = [r for r in rows if "error" not in r]
    summ = {"n": len(ok), "n_failed": len(rows) - len(ok), "requested_model": JEV_MODEL,
            "jev_top1": float(np.mean([r["jev_correct"] for r in ok])) if ok else None,
            "markov_top1_same_rows": float(np.mean([r["markov_correct"] for r in ok])) if ok else None,
            "cost_usd": round(runner.cost, 4), "aborted": runner.abort}
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summ, indent=2))
    runner.progress(f"SUMMARY {json.dumps(summ)}")
    runner.progress("DONE")
    return 2 if runner.abort else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["synth", "meps"], required=True)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cost-cap", type=float, default=12.0)
    ap.add_argument("--n", type=int, default=150, help="meps arm sample size (the Haiku probe used 150)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N tasks only")
    args = ap.parse_args()
    _api_key()  # fail fast, before any retry back-off, if no key is configured
    args.out = args.out or HERE / ("results_jev_n200" if args.arm == "synth" else "results_meps_llm_jev")
    if args.limit:
        orig = Runner.run_tasks
        Runner.run_tasks = lambda self, tasks, fn: orig(self, tasks[: args.limit], fn)
    return arm_synth(args) if args.arm == "synth" else arm_meps(args)


if __name__ == "__main__":
    raise SystemExit(main())
