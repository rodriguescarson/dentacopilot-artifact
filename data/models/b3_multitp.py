"""B3 — MultiTP-style CNN-RNN sequence comparator (Liu et al., AIM 2024).

A minimal reimplementation of the published architecture for direct
comparison: an embedding layer over CDT codes, a 1-D CNN block to integrate
local context (the most recent ~5 procedures, mimicking MultiTP's "tooth +
neighbour" tensor convolution), a GRU to model the longer sequence, and a
single-layer attention head over the GRU outputs whose final pooled state
is mapped to a softmax over the CDT vocabulary.

Where MultiTP differs in scope: it operates on partial-edentulism cases
with a fixed five-treatment output sequence. We adapt the same primitives
to single-step next-procedure prediction over the broader vocabulary so the
comparator is fair against B0–B2 and M1–M3.

Hyperparameters are deliberately small (embed=32, hidden=64, ~30k params)
because our synthetic corpus has only ~6k training examples; bigger nets
overfit visibly. On real clinical EHR data we expect to scale up.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

from .base import Example, Model


class _MultiTPNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        cnn_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.cnn = nn.Conv1d(embed_dim, hidden_dim, kernel_size=cnn_kernel, padding=cnn_kernel // 2)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T) integer code ids; 0 = pad
        emb = self.embed(x)                        # (B, T, E)
        h = self.cnn(emb.transpose(1, 2))          # (B, H, T)
        h = F.relu(h).transpose(1, 2)              # (B, T, H)
        out, _ = self.rnn(h)                       # (B, T, H)
        # Length-masked attention pooling
        mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1))
        scores = self.attn(out).squeeze(-1)        # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * out).sum(dim=1)  # (B, H)
        return self.head(pooled)


class MultiTPModel(Model):
    name = "b3_multitp"

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        max_len: int = 16,
        epochs: int = 12,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        if not _HAVE_TORCH:
            raise RuntimeError("torch is required for B3; pip install torch")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.code_to_id: dict[str, int] = {}
        self.id_to_code: dict[int, str] = {}
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}
        self.net: _MultiTPNet | None = None

    # ----- helpers --------------------------------------------------------
    def _build_vocab(self, train_examples: Sequence[Example]) -> None:
        codes: set[str] = set()
        labels: set[str] = set()
        for ex in train_examples:
            for ev in ex.history:
                codes.add(ev.code)
            labels.add(ex.gold)
        # 0 reserved for pad
        for i, code in enumerate(sorted(codes), start=1):
            self.code_to_id[code] = i
            self.id_to_code[i] = code
        for i, lab in enumerate(sorted(labels)):
            self.label_to_id[lab] = i
            self.id_to_label[i] = lab

    def _encode(self, example: Example) -> tuple[list[int], int]:
        ids = [self.code_to_id.get(ev.code, 0) for ev in example.history[-self.max_len:]]
        length = len(ids)
        ids = ids + [0] * (self.max_len - length)
        return ids, length

    def _batches(
        self,
        examples: Sequence[Example],
        batch_size: int,
        shuffle: bool,
    ):
        n = len(examples)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            chunk = idx[start : start + batch_size]
            seqs, lens, ys = [], [], []
            for k in chunk:
                ex = examples[int(k)]
                ids, ln = self._encode(ex)
                # If an example has 0-length history (shouldn't happen in our
                # expansion, but guard anyway), skip.
                if ln == 0:
                    continue
                seqs.append(ids)
                lens.append(ln)
                ys.append(self.label_to_id.get(ex.gold, -1))
            if not seqs:
                continue
            seq_t = torch.tensor(seqs, dtype=torch.long)
            len_t = torch.tensor(lens, dtype=torch.long)
            y_t = torch.tensor(ys, dtype=torch.long)
            valid = y_t >= 0
            yield seq_t[valid], len_t[valid], y_t[valid]

    # ----- Model interface ------------------------------------------------
    def fit(self, train_examples: Sequence[Example]) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._build_vocab(train_examples)

        self.net = _MultiTPNet(
            vocab_size=len(self.code_to_id),
            n_classes=len(self.label_to_id),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.net.train()
        for ep in range(self.epochs):
            running = 0.0
            n_seen = 0
            for seq, ln, y in self._batches(train_examples, self.batch_size, shuffle=True):
                opt.zero_grad()
                logits = self.net(seq, ln)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                running += float(loss.item()) * y.size(0)
                n_seen += y.size(0)
            if ep == 0 or (ep + 1) % 4 == 0 or ep == self.epochs - 1:
                print(f"   [b3] epoch {ep + 1}/{self.epochs}  loss={running / max(n_seen, 1):.4f}")

    def predict(self, example: Example, top_k: int = 5) -> list[tuple[str, float]]:
        if self.net is None:
            return []
        self.net.eval()
        ids, ln = self._encode(example)
        if ln == 0:
            return []
        with torch.no_grad():
            seq = torch.tensor([ids], dtype=torch.long)
            length = torch.tensor([ln], dtype=torch.long)
            logits = self.net(seq, length)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = np.argsort(-probs)[:top_k]
        return [(self.id_to_label[int(i)], float(probs[int(i)])) for i in top_idx]
