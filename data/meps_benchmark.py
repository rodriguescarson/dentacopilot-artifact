"""MEPS real-data benchmark for DentaCoPilot (category-granularity).

Public, no-DUA real-world validation complementing the synthetic corpus. Uses the
MEPS 2023 Dental Visits event file (AHRQ HC-248B). Task: given a patient's most
recent dental visit category (+ provider type), predict the category of the NEXT
visit. Category granularity (~7 classes), NOT CDT codes -- MEPS does not record CDT.

Categories are derived from per-visit MEPS treatment flags (1=Yes); each visit is
assigned its single most clinically significant procedure (priority order below),
so a visit coded exam+filling is labelled "restorative", not "exam".

Baselines:
  B0  marginal      -- always predict the globally most frequent next category
  B1  Markov/bigram -- predict argmax P(next | current category)
  B2  logistic reg  -- features = current category + provider type

Patient-level train/test split (no leakage). Seed 42.

Run: python3 meps_benchmark.py            (full)
     python3 meps_benchmark.py --smoke     (quick)
"""
from __future__ import annotations
import argparse, json, pathlib, random
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DTA = HERE / "cache" / "meps" / "h248b.dta"
SEED = 42

# per-visit treatment flags (MEPS HC-248B), in priority order (most significant first)
PRIORITY = [
    ("IMPLANTX", "implant"),
    ("ORALSURX", "oral_surgery"),
    ("ROOTCANX", "endodontic"),
    ("GUMSURGX", "periodontal"),
    ("BRIDGESX", "prosthodontic"),
    ("ORTHDONX", "orthodontic"),
    ("FILLINGX", "restorative"),
    ("SEALANTX", "preventive"),
    ("FLUORIDX", "preventive"),
    ("CLENTETX", "prophylaxis"),   # CLEANING/PROPHYLAXIS (correct flag; DENTPROX = "other")
    ("EXAMINEX", "exam"),
    ("JUSTXRYX", "exam"),
]
OTHER = "other_none"


def primary_category(row) -> str:
    for flag, name in PRIORITY:
        if row.get(flag, 2) == 1:
            return name
    return OTHER


def provider_type(row) -> str:
    if row.get("PEDDENT_M18", 2) == 1: return "pediatric"
    if row.get("ORTHDONX", 2) == 1:    return "ortho_provider"
    if row.get("DENTHYG_M18", 2) == 1: return "hygienist"
    if row.get("GENDENT_M18", 2) == 1: return "general"
    return "unknown"


def build_transitions(df: pd.DataFrame):
    df = df.copy()
    df["cat"] = df.apply(primary_category, axis=1)
    df["prov"] = df.apply(provider_type, axis=1)
    # order events within person by (year, month); stable for ties
    df["order"] = df["DVDATEYR"].astype(float) * 12 + df["DVDATEMM"].astype(float)
    df = df.sort_values(["DUPERSID", "order"], kind="stable")
    trans = []  # (dupersid, prev_cat, prev_prov, next_cat)
    for pid, g in df.groupby("DUPERSID", sort=False):
        cats = g["cat"].tolist(); provs = g["prov"].tolist()
        for i in range(len(cats) - 1):
            trans.append((pid, cats[i], provs[i], cats[i + 1]))
    return pd.DataFrame(trans, columns=["pid", "prev_cat", "prev_prov", "next_cat"])


def topk_acc(true, ranked_lists, k):
    return float(np.mean([t in r[:k] for t, r in zip(true, ranked_lists)]))


def macro_f1(true, pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(true, pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(true, pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED)

    df = pd.read_stata(DTA, convert_categoricals=False)
    if args.smoke:
        df = df[df["DUPERSID"].isin(df["DUPERSID"].drop_duplicates().head(400))]
    tr = build_transitions(df)
    labels = sorted(tr["next_cat"].unique())
    print(f"transitions: {len(tr)} | patients: {tr['pid'].nunique()} | classes: {len(labels)}")
    print("class distribution (next_cat):")
    for lab, n in tr["next_cat"].value_counts().items():
        print(f"   {lab:<18} {n:>5} ({100*n/len(tr):.1f}%)")

    # patient-level split
    pids = tr["pid"].drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
    cut = int(0.7 * len(pids)); train_pids = set(pids[:cut])
    train = tr[tr["pid"].isin(train_pids)]; test = tr[~tr["pid"].isin(train_pids)]
    print(f"\ntrain transitions: {len(train)} | test transitions: {len(test)}")

    y_true = test["next_cat"].tolist()
    results = {}

    # B0 marginal
    order = train["next_cat"].value_counts().index.tolist()
    b0_rank = [order for _ in range(len(test))]
    b0_pred = [order[0]] * len(test)
    results["B0_marginal"] = {
        "top1": topk_acc(y_true, b0_rank, 1), "top3": topk_acc(y_true, b0_rank, 3),
        "macro_f1": macro_f1(y_true, b0_pred, labels)}

    # B1 Markov: P(next | prev_cat)
    cond = (train.groupby("prev_cat")["next_cat"].value_counts().groupby(level=0)
            .apply(lambda s: s.index.get_level_values(1).tolist()).to_dict())
    glob = order
    b1_rank = [cond.get(pc, glob) + [x for x in glob if x not in cond.get(pc, glob)]
               for pc in test["prev_cat"]]
    b1_pred = [r[0] for r in b1_rank]
    results["B1_markov"] = {
        "top1": topk_acc(y_true, b1_rank, 1), "top3": topk_acc(y_true, b1_rank, 3),
        "macro_f1": macro_f1(y_true, b1_pred, labels)}

    # B2 logistic regression on (prev_cat, prev_prov)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown="ignore")
        Xtr = enc.fit_transform(train[["prev_cat", "prev_prov"]])
        Xte = enc.transform(test[["prev_cat", "prev_prov"]])
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(Xtr, train["next_cat"])
        proba = clf.predict_proba(Xte); classes = clf.classes_
        rank = [[classes[i] for i in np.argsort(p)[::-1]] for p in proba]
        pred = [r[0] for r in rank]
        results["B2_logreg"] = {
            "top1": topk_acc(y_true, rank, 1), "top3": topk_acc(y_true, rank, 3),
            "macro_f1": macro_f1(y_true, pred, labels)}
    except Exception as e:
        results["B2_logreg"] = {"error": str(e)}

    print("\n=== MEPS 2023 next-category benchmark (test set) ===")
    print(f"{'model':<14}{'top-1':>9}{'top-3':>9}{'macro-F1':>10}")
    for m, r in results.items():
        if "error" in r:
            print(f"{m:<14}  ERROR: {r['error']}"); continue
        print(f"{m:<14}{r['top1']*100:>8.1f}%{r['top3']*100:>8.1f}%{r['macro_f1']:>10.3f}")

    out = HERE / "results_meps_2023"
    out.mkdir(exist_ok=True)
    summary = {"dataset": "MEPS HC-248B (2023 Dental Visits)", "seed": SEED,
               "n_transitions": len(tr), "n_patients": int(tr["pid"].nunique()),
               "n_classes": len(labels), "labels": labels,
               "n_train": len(train), "n_test": len(test),
               "class_distribution": {k: int(v) for k, v in tr["next_cat"].value_counts().items()},
               "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
