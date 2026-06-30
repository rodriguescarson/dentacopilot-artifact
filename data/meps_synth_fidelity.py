"""Face-validity check: does the synthetic generator's category dynamics
resemble real MEPS dental-visit dynamics?

Addresses the reviewer concern that the synthetic corpus is "made up." We map
BOTH the synthetic corpus (CDT codes -> category) and the MEPS 2023 dental file
to a common 8-category scheme, aggregate to one *primary* procedure per visit
(same priority rule on both), build first-order visit-to-visit transitions, and
compare marginal category distributions and transition matrices.

This does NOT claim the generator reproduces US-population statistics (different
granularity + population); it tests whether the generator's category dynamics
are *in the ballpark* of real dental care sequences rather than arbitrary.

Run: python3 meps_synth_fidelity.py
"""
from __future__ import annotations
import json, pathlib, collections, math
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from cdt_codes import category_for  # CDT code -> one of 12 CDT categories

# common scheme aligned to MEPS treatment flags
COMMON = ["exam", "preventive_prophylaxis", "restorative", "endodontic", "periodontal",
          "surgical", "prosthodontic", "orthodontic", "other"]
PRIORITY = ["surgical", "endodontic", "periodontal", "prosthodontic", "orthodontic",
            "restorative", "preventive_prophylaxis", "exam", "other"]  # most significant first

CDT2COMMON = {
    "diagnostic": "exam", "preventive": "preventive_prophylaxis", "restorative": "restorative",
    "endodontic": "endodontic", "periodontic": "periodontal", "implant": "surgical",
    "oral_surgery": "surgical", "prosthodontic_removable": "prosthodontic",
    "prosthodontic_fixed": "prosthodontic", "maxillofacial": "surgical",
    "orthodontic": "orthodontic", "adjunctive": "other",
}


def primary(cats):  # pick most clinically significant common-category present
    s = set(cats)
    for p in PRIORITY:
        if p in s:
            return p
    return "other"


def synth_transitions():
    d = json.load(open(HERE / "cache" / "synth_n500_seed42.json"))
    trans = []
    for ch in d["charts"]:
        byvisit = collections.defaultdict(list)
        for ev in ch["history"]:
            byvisit[ev["date"]].append(CDT2COMMON.get(category_for(ev["code"]), "other"))
        visits = [primary(byvisit[dt]) for dt in sorted(byvisit)]
        trans += list(zip(visits[:-1], visits[1:]))
    return trans


def meps_transitions():
    df = pd.read_stata(HERE / "cache" / "meps" / "h248b.dta", convert_categoricals=False)
    FLAGS = [("IMPLANTX", "surgical"), ("ORALSURX", "surgical"), ("ROOTCANX", "endodontic"),
             ("GUMSURGX", "periodontal"), ("BRIDGESX", "prosthodontic"), ("ORTHDONX", "orthodontic"),
             ("FILLINGX", "restorative"), ("SEALANTX", "preventive_prophylaxis"),
             ("FLUORIDX", "preventive_prophylaxis"), ("CLENTETX", "preventive_prophylaxis"),
             ("EXAMINEX", "exam"), ("JUSTXRYX", "exam")]
    def cat(row):
        for f, n in FLAGS:
            if row.get(f, 2) == 1: return n
        return "other"
    df["cat"] = df.apply(cat, axis=1)
    df["o"] = df["DVDATEYR"].astype(float) * 12 + df["DVDATEMM"].astype(float)
    df = df.sort_values(["DUPERSID", "o"], kind="stable")
    trans = []
    for _, g in df.groupby("DUPERSID", sort=False):
        c = g["cat"].tolist()
        trans += list(zip(c[:-1], c[1:]))
    return trans


def marginal(trans):
    c = collections.Counter(b for _, b in trans); tot = sum(c.values())
    return np.array([c.get(k, 0) / tot for k in COMMON])


def tmatrix(trans):
    idx = {k: i for i, k in enumerate(COMMON)}; M = np.zeros((len(COMMON), len(COMMON)))
    for a, b in trans: M[idx[a], idx[b]] += 1
    rs = M.sum(1, keepdims=True); rs[rs == 0] = 1
    return M / rs, M


def main():
    st, mt = synth_transitions(), meps_transitions()
    sm, mm = marginal(st), marginal(mt)
    sM, sCnt = tmatrix(st); mM, mCnt = tmatrix(mt)
    print(f"synthetic transitions: {len(st)} | MEPS transitions: {len(mt)}\n")
    print("=== Marginal next-category distribution (common scheme) ===")
    print(f"{'category':<16}{'synthetic':>11}{'MEPS':>9}")
    for k, s, m in zip(COMMON, sm, mm):
        print(f"  {k:<14}{s*100:>9.1f}%{m*100:>8.1f}%")
    # marginal agreement
    r_marg = np.corrcoef(sm, mm)[0, 1]
    tvd = 0.5 * np.abs(sm - mm).sum()
    # transition agreement on cells with support in BOTH
    mask = (sCnt.sum(1, keepdims=True) >= 20) & (mCnt.sum(1, keepdims=True) >= 20)
    mask = np.broadcast_to(mask, sM.shape)
    r_trans = np.corrcoef(sM[mask], mM[mask])[0, 1]
    print(f"\n=== Fidelity ===")
    print(f"  marginal Pearson r        = {r_marg:.3f}")
    print(f"  marginal total-variation  = {tvd:.3f}  (0=identical, 1=disjoint)")
    print(f"  transition-prob Pearson r = {r_trans:.3f}  (supported rows only)")
    out = HERE / "results_meps_2023" / "fidelity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "n_synth": len(st), "n_meps": len(mt), "common_scheme": COMMON,
        "marginal_synth": sm.round(4).tolist(), "marginal_meps": mm.round(4).tolist(),
        "marginal_pearson_r": round(float(r_marg), 3), "marginal_tvd": round(float(tvd), 3),
        "transition_pearson_r": round(float(r_trans), 3)}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
