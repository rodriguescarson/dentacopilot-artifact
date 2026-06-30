"""Selective-prediction (abstention) evaluation on in-distribution vs
out-of-distribution / adversarial charts.

The apples-to-apples LLM run produced 0/30 `abstain` emissions -- a vacuous
differentiator. The better signal (per the paper) is a threshold over the
classical model's *calibrated top-1 probability*. Here we show that this rule
actually fires: it abstains on a controlled fraction of in-distribution cases
(reducing risk among the covered) and abstains far more often on OOD /
adversarial charts -- a working safety behaviour.

OOD/adversarial charts (three constructions, each = a real test chart whose
history is corrupted so no archetype transition structure remains):
  * RANDOM   - each prior code replaced by a uniformly random CDT code
  * SCRAMBLE - the chart's own codes randomly permuted (breaks ordering)
  * EMPTY    - history truncated to a single (uninformative) first event

Operating point: threshold tau chosen for 80% coverage on in-distribution test.
Report: in-dist coverage & risk; OOD abstention rate (= OOD-detection recall);
OOD-detection AUROC using top-1 prob as the score; coverage-risk curve.

Run: python3 abstention_ood.py
"""
from __future__ import annotations
import json, pathlib, random
import numpy as np
import sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "models"))
import synth_charts
from models.base import expand_charts, patient_split
from models.b1_tfidf_lr import TfidfLrModel
from cdt_codes import CDT_CODES

SEED = 42


def top1(model, ex):
    p = model.predict(ex, top_k=1)
    return p[0][1] if p else 0.0


def corrupt(examples, mode, rng):
    vocab = list(CDT_CODES.keys())
    out = []
    for ex in examples:
        e = __import__("copy").deepcopy(ex)
        if mode == "random":
            for ev in e.history: ev.code = rng.choice(vocab)
        elif mode == "scramble":
            codes = [ev.code for ev in e.history]; rng.shuffle(codes)
            for ev, c in zip(e.history, codes): ev.code = c
        elif mode == "empty":
            e.history = e.history[:1]
        out.append(e)
    return out


def main():
    rng = random.Random(SEED)
    charts = synth_charts.load(HERE / "cache" / "synth_n500_seed42.json")
    tr, dev, te = patient_split(charts, seed=SEED)
    train_ex = expand_charts(tr); test_ex = expand_charts(te)
    model = TfidfLrModel(); model.fit(train_ex)

    # in-distribution: top-1 prob + correctness
    id_p = np.array([top1(model, e) for e in test_ex])
    id_correct = np.array([model.predict(e, top_k=1)[0][0] == e.gold for e in test_ex])

    # threshold tau for 80% coverage on in-dist (abstain if prob < tau)
    tau = float(np.quantile(id_p, 0.20))
    cov_mask = id_p >= tau
    coverage = cov_mask.mean()
    risk_cov = 1 - id_correct[cov_mask].mean()       # error among covered
    risk_all = 1 - id_correct.mean()                 # error with no abstention
    print(f"in-distribution test examples: {len(test_ex)}")
    print(f"operating tau (80% target coverage) = {tau:.3f}")
    print(f"  in-dist coverage = {coverage*100:.1f}% | risk(covered) = {risk_cov*100:.1f}%  vs  risk(no-abstain) = {risk_all*100:.1f}%")

    print("\n=== OOD / adversarial abstention (at the same tau) ===")
    print(f"{'construction':<12}{'abstain rate':>13}{'OOD-AUROC':>11}")
    rows = {}
    for mode in ("random", "scramble", "empty"):
        ood = corrupt(test_ex, mode, random.Random(SEED))
        ood_p = np.array([top1(model, e) for e in ood])
        abstain_rate = (ood_p < tau).mean()
        # OOD-detection AUROC: lower top-1 prob should flag OOD
        scores = np.concatenate([-id_p, -ood_p]); labels = np.concatenate([np.zeros(len(id_p)), np.ones(len(ood_p))])
        order = np.argsort(scores); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(scores))
        pos = labels == 1; auroc = (ranks[pos].mean() - (pos.sum()-1)/2) / (~pos).sum()
        rows[mode] = {"abstain_rate": float(abstain_rate), "ood_auroc": float(auroc)}
        print(f"  {mode:<10}{abstain_rate*100:>11.1f}%{auroc:>11.3f}")

    # coverage-risk curve on in-dist
    curve = []
    for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        t = float(np.quantile(id_p, q)); m = id_p >= t
        curve.append({"abstain_target": q, "coverage": float(m.mean()), "risk_covered": float(1-id_correct[m].mean())})

    out = HERE / "results_meps_2023" / "abstention_ood.json"
    out.write_text(json.dumps({
        "seed": SEED, "n_test": len(test_ex), "tau": tau,
        "indist": {"coverage": float(coverage), "risk_covered": float(risk_cov), "risk_no_abstain": float(risk_all)},
        "ood": rows, "coverage_risk_curve": curve}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
