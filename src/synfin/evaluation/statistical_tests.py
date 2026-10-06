"""Statistical tests comparing real and synthetic data distributions."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def is_sliding_windows(windows: np.ndarray) -> bool:
    """True if consecutive windows are stride-1 shifts of one series (they share days)."""
    w = np.asarray(windows)
    return len(w) > 1 and bool(np.allclose(w[:-1, 1:], w[1:, :-1]))


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block bootstrap resample of ``n`` ordered items."""
    n_blocks = max(1, int(np.ceil(n / block)))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def ks_test(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Two-sample Kolmogorov-Smirnov test for each feature.

    The statistic D (max CDF gap, in [0, 1]) is an effect size and is what the
    realism score uses. The p-value is only meaningful when rows are roughly
    independent: pass one timestep per window rather than every timestep of
    overlapping windows (with ~10^5 pooled rows every tiny difference "fails").

    Args:
        real: Real data, shape (N, num_features).
        synthetic: Synthetic data, shape (M, num_features).
        feature_names: Optional list of feature names.
        alpha: Significance level for the ``reject`` flag.

    Returns:
        Dict mapping feature name -> {"statistic", "p_value", "reject"}.
    """
    if real.ndim == 1:
        real = real[:, None]
        synthetic = synthetic[:, None]

    results = {}
    for i in range(real.shape[1]):
        name = feature_names[i] if feature_names else f"feature_{i}"
        stat, p_val = ks_2samp(real[:, i], synthetic[:, i])
        results[name] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "reject": bool(p_val < alpha),
        }
    return results


def mmd_rbf(
    real: np.ndarray,
    synthetic: np.ndarray,
    bandwidth: Optional[float] = None,
    max_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Unbiased squared Maximum Mean Discrepancy with an RBF kernel.

    Uses pairwise distance matrices (memory O(N*M), not O(N*M*D)) and the
    median heuristic for the bandwidth unless one is given. Inputs with more
    than ``max_samples`` rows are randomly subsampled.

    Args:
        real: Real data, shape (N, D). Flattened if higher-dim.
        synthetic: Synthetic data, shape (M, D).
        bandwidth: RBF bandwidth; ``None`` = median pairwise distance.
        max_samples: Row cap per input.
        seed: Seed for subsampling.

    Returns:
        Scalar MMD² estimate (can be slightly negative for identical distributions).
    """
    real = real.reshape(len(real), -1).astype(np.float64)
    synthetic = synthetic.reshape(len(synthetic), -1).astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(real) > max_samples:
        real = real[rng.choice(len(real), max_samples, replace=False)]
    if len(synthetic) > max_samples:
        synthetic = synthetic[rng.choice(len(synthetic), max_samples, replace=False)]

    d_xx = cdist(real, real, "sqeuclidean")
    d_yy = cdist(synthetic, synthetic, "sqeuclidean")
    d_xy = cdist(real, synthetic, "sqeuclidean")
    if bandwidth is None:
        bandwidth = float(np.sqrt(np.median(d_xy[d_xy > 0]) / 2.0)) if (d_xy > 0).any() else 1.0
    gamma = 1.0 / (2.0 * bandwidth**2)

    n, m = len(real), len(synthetic)
    k_xx = np.exp(-gamma * d_xx)
    k_yy = np.exp(-gamma * d_yy)
    k_xy = np.exp(-gamma * d_xy)
    term_xx = (k_xx.sum() - n) / (n * (n - 1))
    term_yy = (k_yy.sum() - m) / (m * (m - 1))
    return float(term_xx + term_yy - 2.0 * k_xy.mean())


def pooled_acf(windows: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation pooled *within* windows, never across window boundaries.

    Concatenating windows end-to-end creates spurious jumps at every boundary
    (and, for overlapping real windows, repeats each observation many times),
    which distorts the ACF. Here lagged products are taken inside each window
    and averaged over all windows.

    Args:
        windows: Shape (num_windows, seq_len) or (seq_len,) for a single series.
        max_lag: Largest lag; capped at ``seq_len - 2``.

    Returns:
        ACF values for lags 0..max_lag (length ``max_lag + 1`` after capping).
    """
    x = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    max_lag = min(max_lag, x.shape[1] - 2)
    xc = x - x.mean()
    var = float((xc**2).mean())
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        cov = float((xc[:, lag:] * xc[:, :-lag]).mean())
        acf.append(cov / (var + 1e-12))
    return np.array(acf)


def acf_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_lag: int = 20,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare per-feature autocorrelation functions (pooled within windows).

    Args:
        real: Real windows (N, seq_len, F), or a single series (T, F) / (T,).
        synthetic: Synthetic data in the same layout.
        max_lag: Maximum lag (capped at seq_len - 2).
        feature_names: Optional feature names.

    Returns:
        Dict mapping feature name -> {"real_acf", "synthetic_acf", "mae"}.
    """

    def as_windows(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a[None, :, None]
        if a.ndim == 2:
            return a[None, :, :]
        return a

    real, synthetic = as_windows(real), as_windows(synthetic)
    results = {}
    for i in range(real.shape[-1]):
        name = feature_names[i] if feature_names else f"feature_{i}"
        acf_real = pooled_acf(real[:, :, i], max_lag)
        acf_synth = pooled_acf(synthetic[:, :, i], max_lag)
        n = min(len(acf_real), len(acf_synth))
        results[name] = {
            "real_acf": acf_real,
            "synthetic_acf": acf_synth,
            "mae": float(np.mean(np.abs(acf_real[:n] - acf_synth[:n]))),
        }
    return results


def cross_correlation_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compare contemporaneous cross-feature correlation matrices.

    Args:
        real: Real data, shape (N, num_features).
        synthetic: Synthetic data, shape (M, num_features).

    Returns:
        Dict with "real_corr", "synthetic_corr", "diff", "frobenius_norm".
    """
    real_corr = np.nan_to_num(np.corrcoef(real, rowvar=False))
    synth_corr = np.nan_to_num(np.corrcoef(synthetic, rowvar=False))
    return {
        "real_corr": real_corr,
        "synthetic_corr": synth_corr,
        "diff": np.abs(real_corr - synth_corr),
        "frobenius_norm": float(np.linalg.norm(real_corr - synth_corr, "fro")),
    }
