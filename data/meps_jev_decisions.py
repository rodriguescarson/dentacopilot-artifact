"""Jev 1.13 (TypeSafe System One decision model, via OpenRouter) on the MEPS 2023 benchmark.

Added 2026-10-10. Jev returns a typed choice with a probability for every option instead of
generated text, so unlike the chat LLMs it can be scored with the same calibration and
split-conformal analysis as the classical model (meps_conformal.py).

Task and data are exactly meps_benchmark.py / meps_conformal.py: next-visit category from the
most recent visit's category and provider type, patient-level 70/30 train/test split (seed 42),
test split halved patient-level into calibration and evaluation (seed 43). Jev sees only those two
features, so its output is a function of the 44 distinct (category, provider) states. Each state
is queried REPEATS times per design; the primary analysis uses the first answer per state (what a
single-call deployment would see) and a sensitivity analysis uses the mean over repeats. The
repeats also measure run-to-run stability, which the Jev docs say is not exact.

Designs:
  J0  zero-shot: the state holds the two features only.
  J1  prior-conditioned (the analogue of M5): the state also holds the next-visit category
      frequencies observed after the same history in the TRAINING split, and their count.
B2 (the multinomial logistic regression of meps_conformal.py) is refit here so every contrast is
paired on the same transitions.

Run:  OPENROUTER_ENV_FILE=~/Projects/.env python3 meps_jev_decisions.py   (calls, then analysis)
      python3 meps_jev_decisions.py --analyse-only
Writes results_meps_jev_decisions/{calls.jsonl, answers.json, analysis.json}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from meps_benchmark import DTA, SEED, build_transitions  # noqa: E402
from meps_conformal import ece  # noqa: E402
from openrouter_compute import _api_key  # noqa: E402

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
REPEATS = 5
OUT = HERE / "results_meps_jev_decisions"
TARGETS = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]

CRITERIA = {  # option -> description, faithful to the MEPS HC-248B flags behind each category
    "implant": "Dental implant work.",
    "oral_surgery": "Oral surgery.",
    "endodontic": "Root canal treatment.",
    "periodontal": "Gum (periodontal) surgery.",
    "prosthodontic": "Bridge work.",
    "orthodontic": "Orthodontic treatment (braces).",
    "restorative": "Fillings.",
    "preventive": "Sealants or fluoride treatment.",
    "prophylaxis": "Cleaning / prophylaxis.",
    "exam": "Examination and/or x-rays only.",
    "other_none": "None of the treatments above was recorded for the visit.",
}
CAT_WORDS = {k: v.rstrip(".").lower() for k, v in CRITERIA.items()}
CAT_WORDS["other_none"] = "a visit with none of the listed treatments recorded"
PROV_WORDS = {"general": "general dentist", "hygienist": "dental hygienist",
              "ortho_provider": "orthodontic provider", "pediatric": "pediatric dentist",
              "unknown": "unknown provider type"}
INSTRUCTIONS = (
    "Each dental visit is labelled by its most significant procedure, in this priority order: "
    "implant, oral surgery, root canal, gum surgery, bridge work, orthodontic, fillings, "
    "sealant or fluoride, cleaning, exam or x-rays; otherwise none. Given the patient's most "
    "recent dental visit, which category will the patient's NEXT dental visit be?")
PRIOR_NOTE = (" The state also lists how often each next-visit category followed this same "
              "history (same visit category and provider type) in a training set of real "
              "patient records, and how many such transitions there were.")


def split():
    tr = build_transitions(pd.read_stata(DTA, convert_categoricals=False))
    pids = tr["pid"].drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
    train_pids = set(pids[: int(0.7 * len(pids))])
    train = tr[tr["pid"].isin(train_pids)]
    test = tr[~tr["pid"].isin(train_pids)].copy()
    test_pids = test["pid"].drop_duplicates().sample(frac=1.0, random_state=SEED + 1).tolist()
    cal_pids = set(test_pids[: len(test_pids) // 2])
    test["is_cal"] = test["pid"].isin(cal_pids)
    return tr, train, test


def state_for(design: str, cat: str, prov: str, train: pd.DataFrame) -> dict:
    st = {"most_recent_visit": CAT_WORDS[cat], "most_recent_visit_provider": PROV_WORDS[prov]}
    if design == "J1":
        sub = train[(train["prev_cat"] == cat) & (train["prev_prov"] == prov)]
        if len(sub) == 0:
            sub = train[train["prev_cat"] == cat]
        freq = sub["next_cat"].value_counts(normalize=True)
        st["training_next_visit_frequencies"] = {c: round(float(freq.get(c, 0.0)), 3) for c in CRITERIA}
        st["training_transitions_with_this_history"] = int(len(sub))
    return st


def call(state: dict, design: str, key: str) -> dict:
    body = {"model": MODEL, "state": state,
            "questions": {"next_visit": {"type": "choice",
                                         "instructions": INSTRUCTIONS + (PRIOR_NOTE if design == "J1" else ""),
                                         "criteria": CRITERIA}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 dentacopilot-research"})
    last = None
    for sleep_s in (0, 5, 30, 90):
        time.sleep(sleep_s)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.load(r)
            ans = payload["answers"]["next_visit"]
            return {"request": body, "model": payload.get("model"), "provider": payload.get("provider"),
                    "id": payload.get("id"), "choice": ans["choice"], "probabilities": ans["probabilities"],
                    "confidence": ans.get("confidence"), "cost_usd": (payload.get("usage") or {}).get("cost"),
                    "latency_ms": int((time.perf_counter() - t0) * 1000)}
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            last = exc.read().decode(errors="replace")[:300] if hasattr(exc, "read") else str(exc)
    raise RuntimeError(f"decisions call failed: {last}")


def run_calls(train: pd.DataFrame, states: list[tuple[str, str]]) -> None:
    key = _api_key()
    OUT.mkdir(exist_ok=True)
    jobs = [(d, c, p, r) for d in ("J0", "J1") for c, p in states for r in range(REPEATS)]
    done = set()
    calls_path = OUT / "calls.jsonl"
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            x = json.loads(line); done.add((x["design"], x["prev_cat"], x["prev_prov"], x["repeat"]))
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} calls to make ({len(done)} done)", flush=True)

    def one(job):
        d, c, p, r = job
        res = call(state_for(d, c, p, train), d, key)
        rec = {"design": d, "prev_cat": c, "prev_prov": p, "repeat": r,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **res}
        with calls_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, rec in enumerate(pool.map(one, jobs), 1):
            if i % 40 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {rec['design']} {rec['prev_cat']}/{rec['prev_prov']} -> {rec['choice']}", flush=True)


def selective(conf_cal, conf_ev, corr_ev, pid_ev, rng) -> list[dict]:
    n_cal = len(conf_cal)
    uniq = np.unique(pid_ev)
    idx_of = {u: np.where(pid_ev == u)[0] for u in uniq}
    rows = []
    for c in TARGETS:
        q = min(max(np.ceil((1 - c) * (n_cal + 1)) / n_cal, 0.0), 1.0)
        tau = float(np.quantile(conf_cal, q, method="lower"))
        acc = conf_ev >= tau
        boots = []
        for _ in range(1000):
            s = rng.choice(uniq, size=len(uniq), replace=True)
            boots.append(acc[np.concatenate([idx_of[u] for u in s])].mean())
        rows.append({"target_coverage": c, "tau": round(tau, 4), "achieved_coverage": round(float(acc.mean()), 4),
                     "coverage_ci95": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
                     "selective_risk": round(float(1 - corr_ev[acc].mean()), 4) if acc.any() else None,
                     "n_accepted": int(acc.sum())})
    return rows


def aurc(conf: np.ndarray, corr: np.ndarray) -> float:
    """Area under the risk-coverage curve, (1/n) sum_i risk(top-i most confident), with ties
    broken uniformly at random in expectation (errors spread evenly across a tied block)."""
    n, total, acc_n, acc_err = len(conf), 0.0, 0, 0.0
    for v in np.unique(conf)[::-1]:
        m = conf == v
        size, err = int(m.sum()), float((1 - corr[m]).sum())
        for j in range(1, size + 1):
            total += (acc_err + j * err / size) / (acc_n + j)
        acc_n += size; acc_err += err
    return total / n


def isotonic_top1(conf: np.ndarray, corr: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """Top-label recalibration: isotonic regression of correctness on top-1 confidence, fit on the
    calibration half and applied to every row (monotone, so the abstention ranking is unchanged)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(conf[cal], corr[cal])
    return iso.predict(conf)


def evaluate(name: str, proba: np.ndarray, classes: list[str], test: pd.DataFrame, rng) -> dict:
    y = test["next_cat"].values
    top = np.argsort(-proba, axis=1)
    pred = np.array(classes)[top[:, 0]]
    corr = (pred == y).astype(int)
    conf = proba.max(axis=1)
    cal = test["is_cal"].values
    pid = test["pid"].values
    yi = np.array([classes.index(v) for v in y])
    nll = float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), yi], 1e-6, 1))))
    brier = float(np.mean(np.sum((proba - np.eye(len(classes))[yi]) ** 2, axis=1)))
    ev = ~cal
    # patient-clustered bootstrap CI on full-test top-1
    uniq = np.unique(pid); idx_of = {u: np.where(pid == u)[0] for u in uniq}
    boots = [corr[np.concatenate([idx_of[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])].mean()
             for _ in range(1000)]
    return {"model": name, "n_test": int(len(y)),
            "top1_test": round(float(corr.mean()), 4),
            "top1_test_ci95_patient_bootstrap": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
            "top3_test": round(float(np.mean([y[i] in np.array(classes)[top[i, :3]] for i in range(len(y))])), 4),
            "nll_test": round(nll, 4), "brier_test": round(brier, 4),
            "n_eval": int(ev.sum()), "full_coverage_risk_eval": round(float(1 - corr[ev].mean()), 4),
            "ece_eval": round(ece(conf[ev], corr[ev]), 4),
            **(lambda r: {"ece_eval_after_isotonic_top1": round(ece(r[ev], corr[ev]), 4),
                          "aurc_eval_after_isotonic_top1": round(aurc(r[ev], corr[ev]), 4)})(
                isotonic_top1(conf, corr, cal)),
            "aurc_eval": round(aurc(conf[ev], corr[ev]), 4),
            "mean_top1_prob_eval": round(float(conf[ev].mean()), 4),
            "pred_distribution_test": pd.Series(pred).value_counts().to_dict(),
            "selective": selective(conf[cal], conf[ev], corr[ev], pid[ev], rng)}, corr


def analyse(tr, train, test) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    rng = np.random.default_rng(SEED)
    classes = list(CRITERIA)
    enc = OneHotEncoder(handle_unknown="ignore")
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(enc.fit_transform(train[["prev_cat", "prev_prov"]]), train["next_cat"])
    p_b2 = clf.predict_proba(enc.transform(test[["prev_cat", "prev_prov"]]))
    b2 = np.zeros((len(test), len(classes)))
    for j, c in enumerate(clf.classes_):
        b2[:, classes.index(c)] = p_b2[:, j]
    calls = [json.loads(l) for l in (OUT / "calls.jsonl").read_text().splitlines()]
    out = {"dataset": "MEPS HC-248B (2023 Dental Visits)", "seed": SEED, "jev_model_requested": MODEL,
           "jev_models_served": sorted({c["model"] for c in calls}), "n_calls": len(calls),
           "cost_usd_total": round(sum(c.get("cost_usd") or 0 for c in calls), 5),
           "n_states": int(tr.groupby(["prev_cat", "prev_prov"]).ngroups), "repeats": REPEATS,
           "n_test": int(len(test)), "n_cal": int(test["is_cal"].sum()), "n_eval": int((~test["is_cal"]).sum())}
    res_b2, corr_b2 = evaluate("B2 logistic regression", b2, classes, test, rng)
    out["B2"] = res_b2
    for d in ("J0", "J1"):
        by_state: dict[tuple, list[dict]] = {}
        for c in calls:
            if c["design"] == d:
                by_state.setdefault((c["prev_cat"], c["prev_prov"]), []).append(c)
        for v in by_state.values():
            v.sort(key=lambda c: c["repeat"])
        vec = lambda c: np.array([float(c["probabilities"].get(k, 0.0)) for k in classes])
        stability = []
        for v in by_state.values():
            m = np.stack([vec(c) for c in v])
            stability.append({"top1_agreement": float(np.mean([c["choice"] == v[0]["choice"] for c in v])),
                              "mean_abs_dev_prob": float(np.mean(np.abs(m - m.mean(0))))})
        for variant, agg in (("first_call", lambda v: vec(v[0])), ("mean_of_repeats", lambda v: np.mean([vec(c) for c in v], 0))):
            P = np.stack([agg(by_state[(r.prev_cat, r.prev_prov)]) for r in test.itertuples()])
            P = P / P.sum(1, keepdims=True)
            res, corr = evaluate(f"Jev {d} {variant}", P, classes, test, rng)
            x = int(np.sum((corr_b2 == 1) & (corr == 0))); y_ = int(np.sum((corr_b2 == 0) & (corr == 1)))
            # paired patient-clustered bootstrap of the top-1 difference (Jev - B2)
            pid = test["pid"].values; uniq = np.unique(pid); idx_of = {u: np.where(pid == u)[0] for u in uniq}
            diffs = []
            for _ in range(1000):
                ii = np.concatenate([idx_of[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
                diffs.append(corr[ii].mean() - corr_b2[ii].mean())
            res["vs_B2_top1"] = {"b2_only_correct": x, "jev_only_correct": y_,
                                 "diff": round(float(corr.mean() - corr_b2.mean()), 4),
                                 "diff_ci95_patient_bootstrap": [round(float(z), 4) for z in np.percentile(diffs, [2.5, 97.5])]}
            # paired patient-clustered bootstrap of the AURC difference (Jev - B2) on the eval half
            ev = ~test["is_cal"].values; pe = pid[ev]; ue = np.unique(pe)
            ie = {u: np.where(pe == u)[0] for u in ue}
            cj, kj = P.max(1)[ev], corr[ev]
            cb, kb = b2.max(1)[ev], corr_b2[ev]
            d_aurc = []
            for _ in range(500):
                ii = np.concatenate([ie[u] for u in rng.choice(ue, size=len(ue), replace=True)])
                d_aurc.append(aurc(cj[ii], kj[ii]) - aurc(cb[ii], kb[ii]))
            res["vs_B2_aurc"] = {"diff": round(aurc(cj, kj) - aurc(cb, kb), 4),
                                 "diff_ci95_patient_bootstrap": [round(float(z), 4) for z in np.percentile(d_aurc, [2.5, 97.5])]}
            out[f"{d}_{variant}"] = res
        out[f"{d}_stability"] = {
            "states": len(stability),
            "mean_top1_agreement_across_repeats": round(float(np.mean([s["top1_agreement"] for s in stability])), 4),
            "states_with_any_top1_flip": int(sum(s["top1_agreement"] < 1 for s in stability)),
            "mean_abs_dev_prob": round(float(np.mean([s["mean_abs_dev_prob"] for s in stability])), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    tr, train, test = split()
    states = sorted(set(zip(tr["prev_cat"], tr["prev_prov"])))
    if not args.analyse_only:
        run_calls(train, states)
    out = analyse(tr, train, test)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2))
    show = {k: v for k, v in out.items() if not isinstance(v, dict)}
    print(json.dumps(show, indent=1))
    for k in ("B2", "J0_first_call", "J0_mean_of_repeats", "J1_first_call", "J1_mean_of_repeats"):
        r = out[k]
        sel50 = next(s for s in r["selective"] if s["target_coverage"] == 0.5)
        print(f"{k:22s} top1={r['top1_test']:.3f} {r['top1_test_ci95_patient_bootstrap']} ECE={r['ece_eval']:.3f} "
              f"risk={r['full_coverage_risk_eval']:.3f} -> {sel50['selective_risk']} at cov {sel50['achieved_coverage']} "
              f"AURC={r['aurc_eval']:.4f} NLL={r['nll_test']:.3f} {r.get('vs_B2_top1', '')}")
    print(out["J0_stability"], out["J1_stability"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
