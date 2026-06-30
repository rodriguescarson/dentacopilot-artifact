"""B2 — XGBoost on engineered chart-history features.

Hand-engineered features over chart prefix:
- One-hot of last code + last code's category
- One-hot of the second-to-last code (if present)
- Counts of prior procedures by category (9 categories + unknown)
- Has any extraction in history? Has any RCT? Has any implant?
- Time since last visit (days)
- Patient age band (one-hot) and sex
- Prefix length

We use XGBoost's softprob multiclass classifier so we can return calibrated
top-K with probabilities. Falls back to scikit-learn if XGBoost is missing.
"""

from __future__ import annotations

from typing import Sequence

import datetime as _dt
import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except ImportError:  # pragma: no cover
    _HAVE_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401

from cdt_codes import CATEGORIES, category_for
from .base import Example, Model


_AGE_BANDS = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
_CATEGORIES = list(CATEGORIES.keys()) + ["unknown"]


def _featurize(example: Example) -> dict[str, float]:
    feats: dict[str, float] = {}
    history = example.history

    # Prefix length
    feats["prefix_len"] = float(len(history))

    # Last and second-to-last code one-hots are too high-cardinality for a
    # static featurizer; we instead encode their category and tooth-arch.
    if history:
        last = history[-1]
        feats[f"last_cat_{category_for(last.code)}"] = 1.0
        if last.tooth is not None:
            feats[f"last_tooth_arch_max"] = 1.0 if 1 <= last.tooth <= 16 else 0.0
            feats[f"last_tooth_arch_mand"] = 1.0 if 17 <= last.tooth <= 32 else 0.0
    if len(history) >= 2:
        prev2 = history[-2]
        feats[f"prev2_cat_{category_for(prev2.code)}"] = 1.0

    # Category counts in history
    for cat in _CATEGORIES:
        feats[f"hist_count_{cat}"] = 0.0
    for ev in history:
        feats[f"hist_count_{category_for(ev.code)}"] += 1.0

    # Booleans
    feats["has_extraction"] = 1.0 if any(
        ev.code in {"D7140", "D7210", "D7220", "D7230", "D7240", "D7241", "D7250"} for ev in history
    ) else 0.0
    feats["has_rct"] = 1.0 if any(
        ev.code in {"D3310", "D3320", "D3330", "D3346", "D3347", "D3348"} for ev in history
    ) else 0.0
    feats["has_implant"] = 1.0 if any(
        ev.code in {"D6010", "D6056", "D6057", "D6058", "D6065"} for ev in history
    ) else 0.0
    feats["has_perio_treatment"] = 1.0 if any(
        ev.code in {"D4341", "D4342", "D4346", "D4910", "D4263"} for ev in history
    ) else 0.0
    feats["has_crown"] = 1.0 if any(
        ev.code in {"D2740", "D2750", "D2790", "D2950"} for ev in history
    ) else 0.0
    feats["has_ortho"] = 1.0 if any(category_for(ev.code) == "orthodontic" for ev in history) else 0.0

    # Time since last visit (days)
    if len(history) >= 1:
        try:
            last_date = _dt.date.fromisoformat(history[-1].date)
            today = _dt.date(2026, 1, 1)
            feats["days_since_last"] = float((today - last_date).days)
        except Exception:
            feats["days_since_last"] = 0.0

    # Demographics
    for band in _AGE_BANDS:
        feats[f"age_{band}"] = 1.0 if example.age_band == band else 0.0
    feats["sex_F"] = 1.0 if example.sex == "F" else 0.0
    feats["sex_M"] = 1.0 if example.sex == "M" else 0.0

    # Medical history flags
    feats["mh_t2dm"] = 1.0 if "T2DM" in example.medical_history else 0.0
    feats["mh_htn"] = 1.0 if "HTN" in example.medical_history else 0.0
    feats["mh_smoker"] = 1.0 if "Smoker" in example.medical_history else 0.0

    return feats


class XGBoostModel(Model):
    name = "b2_xgboost"

    def __init__(self, n_estimators: int = 200, max_depth: int = 6, eta: float = 0.1) -> None:
        if not _HAVE_XGB:
            raise RuntimeError("xgboost is required for B2; install per requirements.txt")
        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=eta,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
            eval_metric="mlogloss",
        )
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []

    def _matrix(self, examples: Sequence[Example]) -> np.ndarray:
        rows = [_featurize(ex) for ex in examples]
        if not self.feature_names:
            keys: set[str] = set()
            for r in rows:
                keys.update(r.keys())
            self.feature_names = sorted(keys)
        return np.array(
            [[r.get(k, 0.0) for k in self.feature_names] for r in rows],
            dtype=np.float32,
        )

    def fit(self, train_examples: Sequence[Example]) -> None:
        X = self._matrix(train_examples)
        y_raw = [ex.gold for ex in train_examples]
        y = self.label_encoder.fit_transform(y_raw)
        self.clf.fit(X, y)

    def predict(self, example: Example, top_k: int = 5) -> list[tuple[str, float]]:
        X = self._matrix([example])
        proba = self.clf.predict_proba(X)[0]
        top_idx = np.argsort(-proba)[:top_k]
        out = []
        for i in top_idx:
            label = self.label_encoder.inverse_transform([i])[0]
            out.append((label, float(proba[i])))
        return out
