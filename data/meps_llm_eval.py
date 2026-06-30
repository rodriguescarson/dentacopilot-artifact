"""LLM-on-MEPS evaluation (real-data, category granularity).

Runs an LLM (via cc_compute -> Claude Code) on a sample of MEPS next-visit
transitions and compares to the Markov baseline. The LLM sees only the prior
visit's category and provider type (the features MEPS provides), so this is a
deliberately thin-feature setting: we EXPECT the LLM to roughly match, not beat,
the classical Markov model -- consistent with the paper's thesis that the LLM's
chart-grounded advantage requires rich charts, not coarse category history.

NOTE: must be run where the `claude` CLI executes UNATTENDED. Inside a nested
Claude Code session the call returns an approval prompt rather than an answer;
run this from a normal shell.

Run: python3 meps_llm_eval.py --n 200 [--model anthropic:claude-haiku-4-5]
"""
from __future__ import annotations
import argparse, json, pathlib, random, re, collections
import numpy as np, pandas as pd, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cc_compute

CATS = ["implant", "oral_surgery", "endodontic", "periodontal", "prosthodontic",
        "orthodontic", "restorative", "preventive", "prophylaxis", "exam", "other_none"]
FLAGS = [("IMPLANTX", "implant"), ("ORALSURX", "oral_surgery"), ("ROOTCANX", "endodontic"),
         ("GUMSURGX", "periodontal"), ("BRIDGESX", "prosthodontic"), ("ORTHDONX", "orthodontic"),
         ("FILLINGX", "restorative"), ("SEALANTX", "preventive"), ("FLUORIDX", "preventive"),
         ("CLENTETX", "prophylaxis"), ("EXAMINEX", "exam"), ("JUSTXRYX", "exam")]
BYPASS = ["--permission-mode", "bypassPermissions"]
SEED = 42


def build_transitions():
    df = pd.read_stata(HERE / "cache" / "meps" / "h248b.dta", convert_categoricals=False)
    def cat(r):
        for f, n in FLAGS:
            if r.get(f, 2) == 1: return n
        return "other_none"
    def prov(r):
        if r.get("PEDDENT_M18", 2) == 1: return "pediatric"
        if r.get("DENTHYG_M18", 2) == 1: return "hygienist"
        if r.get("GENDENT_M18", 2) == 1: return "general dentist"
        return "unknown"
    df["cat"] = df.apply(cat, axis=1); df["prov"] = df.apply(prov, axis=1)
    df["o"] = df["DVDATEYR"].astype(float) * 12 + df["DVDATEMM"].astype(float)
    df = df.sort_values(["DUPERSID", "o"], kind="stable")
    tr = []
    for _, g in df.groupby("DUPERSID", sort=False):
        c, p = g["cat"].tolist(), g["prov"].tolist()
        for i in range(len(c) - 1):
            tr.append((c[i], p[i], c[i + 1]))
    return pd.DataFrame(tr, columns=["prev_cat", "prev_prov", "next_cat"])


def prompt_for(prev_cat, prev_prov):
    return (
        "You are predicting the category of a dental patient's NEXT visit.\n"
        f"Categories (choose exactly one): {', '.join(CATS)}.\n"
        f"The patient's most recent visit was a '{prev_cat}' visit with a {prev_prov}.\n"
        "Reply with ONLY the single most likely next-visit category from the list."
    )


def parse(text):
    t = (text or "").lower()
    for c in CATS:
        if c.replace("_", " ") in t or c in t: return c
    return "other_none"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=None); args = ap.parse_args()
    rng = random.Random(SEED)
    tr = build_transitions()
    pids = None
    # Markov baseline trained on 70% (by row, seeded) -> evaluate LLM on a sample of the held-out 30%
    idx = list(range(len(tr))); rng.shuffle(idx); cut = int(0.7 * len(idx))
    train, test = tr.iloc[idx[:cut]], tr.iloc[idx[cut:]]
    cond = (train.groupby("prev_cat")["next_cat"].agg(lambda s: s.value_counts().idxmax())).to_dict()
    glob = train["next_cat"].value_counts().idxmax()
    sample = test.sample(n=min(args.n, len(test)), random_state=SEED)
    run_dir = HERE / "results_meps_llm"; run_dir.mkdir(exist_ok=True)
    log = cc_compute.JsonlLogger(run_dir) if hasattr(cc_compute, "JsonlLogger") else None
    llm_correct = markov_correct = 0
    for _, r in sample.iterrows():
        kw = {"prompt": prompt_for(r["prev_cat"], r["prev_prov"]), "model": args.model, "extra_args": BYPASS}
        if log: kw["logger"] = log
        res = cc_compute.cc_call(**kw)
        pred = parse(getattr(res, "text", str(res)))
        mk = cond.get(r["prev_cat"], glob)
        llm_correct += (pred == r["next_cat"]); markov_correct += (mk == r["next_cat"])
    n = len(sample)
    summary = {"n": n, "model": cc_compute._resolve_model(args.model),
               "llm_top1": llm_correct / n, "markov_top1": markov_correct / n}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nExpectation: LLM ~= Markov on these thin features (no rich chart to ground on).")


if __name__ == "__main__":
    main()
