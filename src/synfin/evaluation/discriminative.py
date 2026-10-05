"""Discriminative score: can a classifier tell real windows from synthetic ones?

A classifier two-sample test in the spirit of TimeGAN's "discriminative
score", but on per-window *dynamics* features (volatility level, tail weight,
autocorrelation of |x|, leverage) rather than raw values. KS and MMD on raw or
flattened windows are close to blind to temporal structure: i.i.d. noise with
the right mean and variance passes them. These features are exactly where that
noise differs from real returns.

The classifier is scored with k-fold cross-validated ROC AUC. Folds are
contiguous blocks of each class, so the near-duplicate neighbours that
overlapping (stride-1) real windows produce stay in the same fold instead of
leaking between train and test.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def _acf1(x: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each row of ``x`` (shape (N, T))."""
    xc = x - x.mean(axis=1, keepdims=True)
    den = (xc**2).sum(axis=1) + 1e-12
    return (xc[:, 1:] * xc[:, :-1]).sum(axis=1) / den


def window_features(windows: np.ndarray, return_idx: Optional[int] = None) -> np.ndarray:
    """Per-window summary features, shape (N, n_features).

    For every channel: mean, log std, lag-1 ACF and lag-1 ACF of |x - mean|.
    For the return channel additionally: kurtosis, max |r| / std, and the
    leverage correlation corr(r_t, |r_{t+1}|).
    """
    feats = []
    for c in range(windows.shape[-1]):
        x = windows[:, :, c].astype(np.float64)
        dev = np.abs(x - x.mean(axis=1, keepdims=True))
        feats += [x.mean(axis=1), np.log(x.std(axis=1) + 1e-12), _acf1(x), _acf1(dev)]
    if return_idx is not None:
        r = windows[:, :, return_idx].astype(np.float64)
        rc = r - r.mean(axis=1, keepdims=True)
        sd = rc.std(axis=1) + 1e-12
        feats.append((rc**4).mean(axis=1) / sd**4)
        feats.append(np.abs(rc).max(axis=1) / sd)
        a, b = rc[:, :-1], np.abs(r[:, 1:])
        b = b - b.mean(axis=1, keepdims=True)
        lev = (a * b).sum(axis=1) / (np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)) + 1e-12)
        feats.append(lev)
    return np.column_stack(feats)


def discriminative_score(
    real: np.ndarray,
    synthetic: np.ndarray,
    return_idx: Optional[int] = None,
    n_folds: int = 5,
    max_per_class: int = 3000,
    seed: int = 0,
) -> Dict[str, float]:
    """Cross-validated AUC of a real-vs-synthetic classifier on window features.

    Args:
        real: Real windows (N, seq_len, F), in time order.
        synthetic: Synthetic windows (M, seq_len, F).
        return_idx: Index of the log-return channel (adds tail/leverage features).
        n_folds: Number of contiguous-block CV folds.
        max_per_class: Cap per class (evenly spaced subsample, keeps time order).
        seed: Classifier seed.

    Returns:
        Dict with ``auc`` (0.5 = indistinguishable, 1.0 = trivially separable)
        and ``score`` = 1 - 2 * max(0, auc - 0.5), in [0, 1].
    """
    n = min(len(real), len(synthetic), max_per_class)
    if n < 2 * n_folds:
        raise ValueError("Too few windows for the discriminative score.")
    real = real[np.linspace(0, len(real) - 1, n).astype(int)]
    synthetic = synthetic[np.linspace(0, len(synthetic) - 1, n).astype(int)]

    X = np.vstack([window_features(real, return_idx), window_features(synthetic, return_idx)])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.r_[np.ones(n), np.zeros(n)]
    block = np.arange(n) * n_folds // n
    folds = np.r_[block, block]

    proba = np.empty(2 * n)
    for k in range(n_folds):
        test = folds == k
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=seed)
        clf.fit(X[~test], y[~test])
        proba[test] = clf.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, proba))
    return {"auc": auc, "score": float(np.clip(1.0 - 2.0 * (auc - 0.5), 0.0, 1.0))}
