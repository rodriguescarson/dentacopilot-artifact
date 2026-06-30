"""B1 — TF-IDF over chart history + multinomial logistic regression.

We treat each chart prefix as a "document" whose tokens are the CDT codes
seen so far (optionally with their position-from-end). TF-IDF weighting is
applied across the training corpus, and a multinomial logistic regression
predicts the next code.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .base import Example, Model


def _tokenize(example: Example) -> list[str]:
    """Tokens used by the TF-IDF vectorizer.

    We include the bare code, plus a positional-from-end token for the last
    three procedures so the model can learn locally-recent bigram-ish signal
    on top of bag-of-codes.
    """
    tokens = [ev.code for ev in example.history]
    last_three = list(reversed(example.history[-3:]))
    for offset, ev in enumerate(last_three):
        tokens.append(f"prev{offset}_{ev.code}")
    if example.history:
        tokens.append(f"last_{example.history[-1].code}")
    tokens.append(f"age_{example.age_band}")
    tokens.append(f"sex_{example.sex}")
    return tokens


class TfidfLrModel(Model):
    name = "b1_tfidf_lr"

    def __init__(self, C: float = 1.0, max_iter: int = 200) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=lambda x: x,  # accept pre-tokenized input
            preprocessor=lambda x: x,
            lowercase=False,
            min_df=1,
        )
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=1,
        )
        self.label_encoder = LabelEncoder()

    def fit(self, train_examples: Sequence[Example]) -> None:
        toks = [_tokenize(ex) for ex in train_examples]
        labels = [ex.gold for ex in train_examples]
        X = self.vectorizer.fit_transform(toks)
        y = self.label_encoder.fit_transform(labels)
        self.model.fit(X, y)

    def predict(self, example: Example, top_k: int = 5) -> list[tuple[str, float]]:
        X = self.vectorizer.transform([_tokenize(example)])
        proba = self.model.predict_proba(X)[0]
        top_idx = np.argsort(-proba)[:top_k]
        out = []
        for i in top_idx:
            label = self.label_encoder.inverse_transform([i])[0]
            out.append((label, float(proba[i])))
        return out
