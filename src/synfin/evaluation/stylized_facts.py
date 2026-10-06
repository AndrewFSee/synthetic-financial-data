"""Check whether data reproduces the stylized facts of financial returns.

All checks take *unscaled* log returns. Min-max scaling shifts returns into
[0, 1], after which ``|r|`` is no longer a volatility proxy, so run these on
the original return units (``compute_all_metrics`` does this when given the
feature names).

Inputs may be one series (shape (T,)) or a batch of windows (shape (N, T)).
Lagged statistics are pooled within windows (see
:func:`~synfin.evaluation.statistical_tests.pooled_acf`) so that window
boundaries and overlapping windows do not create artificial dependence.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats

from synfin.evaluation.statistical_tests import pooled_acf


def check_fat_tails(returns: np.ndarray) -> Dict[str, float]:
    """Tail heaviness of returns; daily equity returns have excess kurtosis > 1.

    Raw kurtosis is dominated by a handful of extreme values (for AAPL daily
    returns, dropping the largest 0.1% cuts it from ~6.6 to ~4.7), so robust
    measures are reported alongside it: quantiles of |r| in standard
    deviations, kurtosis with the largest 0.1% of |r| trimmed, and the single
    largest move in standard deviations. Compare those when judging tails.

    Args:
        returns: Log returns, shape (T,) or (N, T).

    Returns:
        Dict with ``kurtosis`` (Pearson), ``excess_kurtosis``,
        ``trimmed_excess_kurtosis``, ``q99_sd``, ``q999_sd``, ``max_sd`` and
        ``is_fat_tailed``.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    excess = float(stats.kurtosis(r, fisher=True))
    dev = np.abs(r - r.mean()) / (r.std() + 1e-12)
    keep = dev < np.percentile(dev, 99.9)
    return {
        "kurtosis": excess + 3.0,
        "excess_kurtosis": excess,
        "trimmed_excess_kurtosis": float(stats.kurtosis(r[keep], fisher=True)),
        "q99_sd": float(np.percentile(dev, 99)),
        "q999_sd": float(np.percentile(dev, 99.9)),
        "max_sd": float(dev.max()),
        "is_fat_tailed": excess > 1.0,
    }


def check_volatility_clustering(returns: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
    """Volatility clustering: positive autocorrelation of |r_t| and r_t^2.

    Args:
        returns: Log returns, shape (T,) or (N, T).
        max_lag: Maximum lag (capped at window length - 2).

    Returns:
        Dict with the mean ACF of |r| and r^2 over lags 1..max_lag, and lag-1 values.
    """
    r = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    abs_acf = pooled_acf(np.abs(r), max_lag)
    sq_acf = pooled_acf(r**2, max_lag)
    return {
        "mean_abs_return_acf": float(abs_acf[1:].mean()),
        "mean_sq_return_acf": float(sq_acf[1:].mean()),
        "abs_return_acf_lag1": float(abs_acf[1]),
        "has_clustering": bool(abs_acf[1:].mean() > 0.05),
    }


def check_leverage_effect(
    returns: np.ndarray,
    lags: int = 5,
    threshold: float = -0.02,
) -> Dict[str, float]:
    """Leverage effect: negative correlation between r_t and future |r_{t+k}|.

    Args:
        returns: Log returns, shape (T,) or (N, T).
        lags: Number of forward lags (capped at window length - 1).
        threshold: Mean correlation below which the effect is flagged.

    Returns:
        Dict with the correlation at each lag and their mean.
    """
    r = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    vol = np.abs(r)
    results: Dict[str, float] = {}
    for lag in range(1, min(lags, r.shape[1] - 1) + 1):
        a = r[:, :-lag].ravel()
        b = vol[:, lag:].ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        results[f"lag_{lag}"] = corr
    vals = [v for v in results.values() if not np.isnan(v)]
    results["mean_leverage_corr"] = float(np.mean(vals)) if vals else 0.0
    results["has_leverage_effect"] = results["mean_leverage_corr"] < threshold
    return results


def check_volume_volatility_correlation(
    volume: np.ndarray,
    returns: np.ndarray,
) -> Dict[str, float]:
    """Volume-volatility correlation: corr(volume_t, |r_t|) is typically positive.

    Args:
        volume: Volume feature (any monotone transform), same shape as returns.
        returns: Log returns.

    Returns:
        Dict with Pearson and Spearman correlations.
    """
    v = np.asarray(volume, dtype=np.float64).ravel()
    a = np.abs(np.asarray(returns, dtype=np.float64).ravel())
    if v.std() == 0 or a.std() == 0:
        return {
            "pearson_correlation": 0.0,
            "spearman_correlation": 0.0,
            "has_vol_volume_corr": False,
        }
    pearson_corr, pearson_p = stats.pearsonr(v, a)
    spearman_corr, spearman_p = stats.spearmanr(v, a)
    return {
        "pearson_correlation": float(pearson_corr),
        "pearson_p_value": float(pearson_p),
        "spearman_correlation": float(spearman_corr),
        "spearman_p_value": float(spearman_p),
        "has_vol_volume_corr": bool(spearman_corr > 0.1),
    }


def check_all_stylized_facts(
    returns: np.ndarray,
    volume: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Run all stylized-fact checks.

    Args:
        returns: Unscaled log returns, shape (T,) or (N, T).
        volume: Optional volume feature with the same shape.

    Returns:
        Nested dict with results for each stylized fact.
    """
    out = {
        "fat_tails": check_fat_tails(returns),
        "volatility_clustering": check_volatility_clustering(returns),
        "leverage_effect": check_leverage_effect(returns),
    }
    if volume is not None:
        out["volume_volatility_correlation"] = check_volume_volatility_correlation(volume, returns)
    return out
