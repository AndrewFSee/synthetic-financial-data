"""Post-hoc RNN metrics from the TimeGAN paper (Yoon et al., NeurIPS 2019, Section 5.2).

* **Discriminative score**: an LSTM trained on raw windows to tell real from
  synthetic. Reported as test accuracy, ``|accuracy - 0.5|`` (0 =
  indistinguishable, 0.5 = trivially separable) and ROC AUC. Unlike the
  feature-based check in :mod:`synfin.evaluation.discriminative`, the network
  learns its own features, so it can catch failures hand-picked features miss.
* **Predictive score**: an LSTM trained to predict every feature one step ahead,
  trained on synthetic (TSTR) or on real training windows (TRTR) and both
  tested on the real holdout. Reported as MAEs and the ratio TSTR / TRTR
  (1.0 = the synthetic data teaches the model as well as real data).

Two deliberate differences from common implementations:

1. **Leak-free discriminator split.** Real windows are usually stride-1
   sliding windows, so a random train/test shuffle puts near-copies of each test
   window in the training set. A single contiguous test block at the end has
   the opposite problem: it tests on a different market regime than the one
   trained on, so real test windows look "unlike real" and AUCs can fall far
   below 0.5. Real windows are therefore cut into short blocks spread over the
   whole period; ~``test_frac`` of the blocks are drawn as test, and training
   windows within one window length of a test block are purged. Synthetic
   windows (independent) are split at random.
2. **Real-data baseline for the predictive score.** On near-white-noise returns
   most of the next step is unpredictable, so the MAE ratio sits near 1.0 for
   almost any generator; read it as a sanity check rather than a fine-grained
   score.

Inputs are standardized per feature with the real data's mean/std.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from synfin.evaluation.statistical_tests import is_sliding_windows

logger = logging.getLogger(__name__)


class _LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


class _LSTMPredictor(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h)


def _standardize(real: np.ndarray, *others: np.ndarray):
    flat = real.reshape(-1, real.shape[-1])
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-8
    return [((x - mu) / sd).astype(np.float32) for x in (real, *others)]


def _real_split(n: int, seq_len: int, sliding: bool, test_frac: float, rng: np.random.Generator):
    """Train/test indices for real windows: purged random blocks if sliding, else random."""
    if not sliding:
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * test_frac)))
        return np.sort(perm[n_test:]), np.sort(perm[:n_test])
    block = 2 * seq_len
    n_blocks = max(2, int(np.ceil(n / block)))
    test_blocks = rng.choice(n_blocks, max(1, int(round(n_blocks * test_frac))), replace=False)
    is_test = np.isin(np.arange(n) // block, test_blocks)
    test = np.flatnonzero(is_test)
    # Purge training windows that share any day with a test window.
    dist = np.abs(np.arange(n)[:, None] - test[None, :]).min(axis=1)
    train = np.flatnonzero(~is_test & (dist >= seq_len))
    return train, test


def rnn_discriminative_score(
    real: np.ndarray,
    synthetic: np.ndarray,
    hidden_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:
    """Train an LSTM to tell real from synthetic windows.

    Args:
        real: Real windows (N, T, F); time-ordered (may overlap).
        synthetic: Synthetic windows (M, T, F).
        hidden_dim: LSTM hidden size.
        num_layers: LSTM layers.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Adam learning rate.
        test_frac: Test fraction (per class).
        seed: Random seed.

    Returns:
        Dict with ``accuracy``, ``score`` = |accuracy - 0.5|, ``auc`` and the
        test-set sizes.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    real_s, synth_s = _standardize(real, synthetic)
    tr_r, te_r = _real_split(
        len(real_s), real_s.shape[1], is_sliding_windows(real[:, :, 0]), test_frac, rng
    )
    perm = rng.permutation(len(synth_s))
    n_te_s = max(1, int(round(len(synth_s) * test_frac)))
    te_s, tr_s = perm[:n_te_s], perm[n_te_s:]
    # Balance classes in both splits.
    n_tr, n_te = min(len(tr_r), len(tr_s)), min(len(te_r), len(te_s))
    tr_r, te_r = rng.choice(tr_r, n_tr, replace=False), rng.choice(te_r, n_te, replace=False)
    tr_s, te_s = tr_s[:n_tr], te_s[:n_te]

    X_tr = np.concatenate([real_s[tr_r], synth_s[tr_s]])
    y_tr = np.r_[np.ones(n_tr), np.zeros(n_tr)].astype(np.float32)
    X_te = np.concatenate([real_s[te_r], synth_s[te_s]])
    y_te = np.r_[np.ones(n_te), np.zeros(n_te)]

    model = _LSTMClassifier(real_s.shape[-1], hidden_dim, num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_te))).numpy()
    acc = float(accuracy_score(y_te, prob > 0.5))
    return {
        "accuracy": acc,
        "score": abs(acc - 0.5),
        "auc": float(roc_auc_score(y_te, prob)),
        "n_test_per_class": int(n_te),
    }


def rnn_predictive_score(
    real_train: np.ndarray,
    synthetic: np.ndarray,
    real_holdout: np.ndarray,
    hidden_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict[str, float]:
    """Next-step prediction MAE: trained on synthetic (TSTR) vs real (TRTR), tested on real.

    Args:
        real_train: Real training windows (N, T, F).
        synthetic: Synthetic windows (M, T, F).
        real_holdout: Real windows never seen by the generator (K, T, F).
        hidden_dim: LSTM hidden size.
        num_layers: LSTM layers.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Adam learning rate.
        seed: Random seed.

    Returns:
        Dict with ``mae_tstr``, ``mae_trtr``, ``ratio`` (TSTR / TRTR) and
        ``mae_naive`` (predicting the training mean, for scale).
    """
    real_s, synth_s, hold_s = _standardize(real_train, synthetic, real_holdout)

    def fit(train: np.ndarray) -> _LSTMPredictor:
        torch.manual_seed(seed)
        model = _LSTMPredictor(train.shape[-1], hidden_dim, num_layers)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train[:, :-1]), torch.from_numpy(train[:, 1:])),
            batch_size=batch_size,
            shuffle=True,
        )
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                nn.functional.l1_loss(model(xb), yb).backward()
                opt.step()
        return model.eval()

    def mae(model: _LSTMPredictor) -> float:
        with torch.no_grad():
            pred = model(torch.from_numpy(hold_s[:, :-1])).numpy()
        return float(np.abs(pred - hold_s[:, 1:]).mean())

    mae_tstr, mae_trtr = mae(fit(synth_s)), mae(fit(real_s))
    naive = float(np.abs(hold_s[:, 1:] - real_s.reshape(-1, real_s.shape[-1]).mean(0)).mean())
    return {
        "mae_tstr": mae_tstr,
        "mae_trtr": mae_trtr,
        "ratio": mae_tstr / mae_trtr if mae_trtr > 0 else float("inf"),
        "mae_naive": naive,
    }


def posthoc_rnn_suite(
    real_train: np.ndarray,
    synthetic: np.ndarray,
    real_holdout: Optional[np.ndarray] = None,
    seed: int = 0,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """Run the discriminative score and, if a holdout is given, the predictive score."""
    out = {"discriminative": rnn_discriminative_score(real_train, synthetic, seed=seed, **kwargs)}
    if real_holdout is not None and len(real_holdout):
        out["predictive"] = rnn_predictive_score(
            real_train, synthetic, real_holdout, seed=seed, **kwargs
        )
    return out
