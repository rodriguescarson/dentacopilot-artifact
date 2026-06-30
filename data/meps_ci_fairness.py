"""Bootstrap confidence intervals + fairness/subgroup analysis for the MEPS
next-category benchmark. Closes TRIPOD+AI items 23a (CIs) and 14/3c (fairness).

CIs: 1000-sample patient-clustered bootstrap of the B1 Markov model on the test split
(transitions are clustered within patients, so we resample patients, not rows).
Fairness: merge MEPS 2023 Full-Year-Consolidated demographics (h251) by DUPERSID
and report B1 top-1 accuracy by sex, age band, race/ethnicity, and poverty
category, with the max-min gap per dimension.

Run: python3 meps_ci_fairness.py
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from meps_benchmark import build_transitions, topk_acc, macro_f1, SEED

FYC = HERE / "cache" / "meps" / "h251.dta"
DENTAL = HERE / "cache" / "meps" / "h248b.dta"

RACETHX = {1: "Hispanic", 2: "NH White", 3: "NH Black", 4: "NH Asian", 5: "NH Other/Multiple"}
POVCAT = {1: "Poor", 2: "Near-poor", 3: "Low-income", 4: "Middle-income", 5: "High-income"}
SEXL = {1: "Male", 2: "Female"}


def age_band(a):
    a = float(a)
    return "0-17" if a < 18 else "18-34" if a < 35 else "35-49" if a < 50 else "50-64" if a < 65 else "65+"


def main():
    rng = np.random.default_rng(SEED)
    tr = build_transitions(pd.read_stata(DENTAL, convert_categoricals=False))
    labels = sorted(tr["next_cat"].unique())
    pids = tr["pid"].drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
    cut = int(0.7 * len(pids)); train_pids = set(pids[:cut])
    train = tr[tr["pid"].isin(train_pids)]; test = tr[~tr["pid"].isin(train_pids)].reset_index(drop=True)

    # B1 Markov
    cond = (train.groupby("prev_cat")["next_cat"].value_counts().groupby(level=0)
            .apply(lambda s: s.index.get_level_values(1).tolist()).to_dict())
    glob = train["next_cat"].value_counts().index.tolist()
    rank = [cond.get(pc, glob) + [x for x in glob if x not in cond.get(pc, glob)] for pc in test["prev_cat"]]
    pred = np.array([r[0] for r in rank]); true = test["next_cat"].to_numpy()
    correct1 = pred == true
    correct3 = np.array([t in r[:3] for t, r in zip(true, rank)])

    # ---- patient-clustered bootstrap 95% CIs (resample PATIENTS, not rows) ----
    # Transitions are clustered within patients (11,016 rows / 5,088 patients), so a
    # row bootstrap is anti-conservative. We resample patient ids with replacement and
    # concatenate their rows, matching meps_conformal.py's cluster bootstrap.
    n = len(test); B = 1000
    pid_arr = test["pid"].to_numpy()
    uniq = np.unique(pid_arr)
    pid_to_idx = {u: np.where(pid_arr == u)[0] for u in uniq}
    def boot(metric):
        vals = []
        for _ in range(B):
            samp = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([pid_to_idx[u] for u in samp])
            vals.append(metric(idx))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    t1 = correct1.mean(); t3 = correct3.mean(); mf1 = macro_f1(true.tolist(), pred.tolist(), labels)
    ci1 = boot(lambda i: correct1[i].mean())
    ci3 = boot(lambda i: correct3[i].mean())
    cif = boot(lambda i: macro_f1(true[i].tolist(), pred[i].tolist(), labels))
    print("=== B1 Markov on MEPS test (95% patient-clustered bootstrap CIs, B=1000) ===")
    print(f"  top-1   {t1*100:.1f}%  [{ci1[0]*100:.1f}, {ci1[1]*100:.1f}]")
    print(f"  top-3   {t3*100:.1f}%  [{ci3[0]*100:.1f}, {ci3[1]*100:.1f}]")
    print(f"  macroF1 {mf1:.3f}  [{cif[0]:.3f}, {cif[1]:.3f}]")

    # ---- fairness: merge demographics by DUPERSID ----
    fyc = pd.read_stata(FYC, convert_categoricals=False)[["DUPERSID", "SEX", "AGELAST", "RACETHX", "POVCAT23"]]
    m = test.copy(); m["correct1"] = correct1
    m = m.merge(fyc, left_on="pid", right_on="DUPERSID", how="left")
    m["sex"] = m["SEX"].map(SEXL); m["age"] = m["AGELAST"].map(age_band)
    m["race"] = m["RACETHX"].map(RACETHX); m["pov"] = m["POVCAT23"].map(POVCAT)
    print(f"\n=== Fairness: B1 top-1 accuracy by subgroup (merged n={m['DUPERSID'].notna().sum()}/{len(m)}) ===")
    # patient-clustered bootstrap of the (uncorrected) max-minus-min gap per dimension
    m_pid = m["pid"].to_numpy()
    muniq = np.unique(m_pid); mpid_to_idx = {u: np.where(m_pid == u)[0] for u in muniq}
    def gap_ci(col):
        valid = m[col].notna().to_numpy()
        vals = []
        for _ in range(B):
            samp = rng.choice(muniq, size=len(muniq), replace=True)
            idx = np.concatenate([mpid_to_idx[u] for u in samp])
            idv = idx[valid[idx]]
            if len(idv) == 0:
                continue
            gg = m.iloc[idv].groupby(col)["correct1"].mean()
            vals.append(gg.max() - gg.min())
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    fairness = {}
    for dim, col in [("Sex", "sex"), ("Age band", "age"), ("Race/ethnicity", "race"), ("Poverty", "pov")]:
        g = m.dropna(subset=[col]).groupby(col)["correct1"].agg(["mean", "count"])
        gap = g["mean"].max() - g["mean"].min()
        glo, ghi = gap_ci(col)
        fairness[dim] = {"groups": {k: [round(v["mean"], 3), int(v["count"])] for k, v in g.iterrows()},
                         "max_min_gap": round(float(gap), 3),
                         "max_min_gap_ci": [round(glo, 3), round(ghi, 3)]}
        print(f"\n  {dim}  (max-min gap = {gap*100:.1f} pp, 95% patient-clustered CI [{glo*100:.1f}, {ghi*100:.1f}])")
        for k, v in g.sort_values("mean", ascending=False).iterrows():
            print(f"    {str(k):<18} {v['mean']*100:5.1f}%  (n={int(v['count'])})")

    out = HERE / "results_meps_2023" / "ci_fairness.json"
    out.write_text(json.dumps({
        "B": B, "n_test": n,
        "b1": {"top1": round(t1, 4), "top1_ci": [round(ci1[0], 4), round(ci1[1], 4)],
               "top3": round(t3, 4), "top3_ci": [round(ci3[0], 4), round(ci3[1], 4)],
               "macro_f1": round(mf1, 4), "macro_f1_ci": [round(cif[0], 4), round(cif[1], 4)]},
        "fairness": fairness}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
