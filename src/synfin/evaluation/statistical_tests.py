"""Statistical tests comparing real and synthetic data distributions."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def ks_test(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Kolmogorov-Smirnov test for each feature.

    Args:
        real: Real data, shape (N, num_features).
        synthetic: Synthetic data, shape (M, num_features).
        feature_names: Optional list of feature names.

    Returns:
        Dict mapping feature name → {"statistic": ..., "p_value": ...}.
    """
    num_features = real.shape[1] if real.ndim > 1 else 1
    if real.ndim == 1:
        real = real[:, None]
        synthetic = synthetic[:, None]

    results = {}
    for i in range(num_features):
        name = feature_names[i] if feature_names else f"feature_{i}"
        stat, p_val = ks_2samp(real[:, i], synthetic[:, i])
        results[name] = {"statistic": float(stat), "p_value": float(p_val)}

    return results


def mmd_rbf(
    real: np.ndarray,
    synthetic: np.ndarray,
    bandwidth: float = 1.0,
) -> float:
    """Maximum Mean Discrepancy (MMD) with RBF kernel.

    Args:
        real: Real data, shape (N, D). Flattened if higher-dim.
        synthetic: Synthetic data, shape (M, D).
        bandwidth: RBF kernel bandwidth.

    Returns:
        Scalar MMD² estimate.
    """
    if real.ndim > 2:
        real = real.reshape(len(real), -1)
    if synthetic.ndim > 2:
        synthetic = synthetic.reshape(len(synthetic), -1)

    def rbf_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - y[None, :, :]
        dist_sq = (diff ** 2).sum(axis=-1)
        return np.exp(-dist_sq / (2 * bandwidth ** 2))

    k_xx = rbf_kernel(real, real).mean()
    k_yy = rbf_kernel(synthetic, synthetic).mean()
    k_xy = rbf_kernel(real, synthetic).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def acf_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_lag: int = 20,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare autocorrelation functions between real and synthetic data.

    Args:
        real: Real data array, shape (N, num_features) or (N,).
        synthetic: Synthetic data array, same shape.
        max_lag: Maximum number of lags to compute.
        feature_names: Optional feature names.

    Returns:
        Dict mapping feature name → {"real_acf": ..., "synthetic_acf": ...}.
    """
    if real.ndim == 1:
        real = real[:, None]
        synthetic = synthetic[:, None]

    num_features = real.shape[1]
    results = {}

    for i in range(num_features):
        name = feature_names[i] if feature_names else f"feature_{i}"
        r = real[:, i]
        s = synthetic[:, i]

        acf_real = _compute_acf(r, max_lag)
        acf_synth = _compute_acf(s, max_lag)
        results[name] = {
            "real_acf": acf_real,
            "synthetic_acf": acf_synth,
            "mae": float(np.mean(np.abs(acf_real - acf_synth))),
        }

    return results


def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to max_lag."""
    n = len(x)
    x_centered = x - x.mean()
    variance = np.var(x)
    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            cov = np.dot(x_centered[lag:], x_centered[:-lag]) / (n - lag)
            acf_vals.append(cov / (variance + 1e-10))
    return np.array(acf_vals)


def cross_correlation_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compare cross-correlation matrices.

    Args:
        real: Real data, shape (N, num_features).
        synthetic: Synthetic data, shape (M, num_features).

    Returns:
        Dict with "real_corr", "synthetic_corr", "diff".
    """
    real_corr = np.corrcoef(real, rowvar=False)
    synth_corr = np.corrcoef(synthetic, rowvar=False)
    return {
        "real_corr": real_corr,
        "synthetic_corr": synth_corr,
        "diff": np.abs(real_corr - synth_corr),
        "frobenius_norm": float(np.linalg.norm(real_corr - synth_corr, "fro")),
    }
