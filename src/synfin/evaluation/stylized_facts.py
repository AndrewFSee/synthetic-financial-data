"""Check whether synthetic data reproduces stylized facts of financial returns."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def check_fat_tails(returns: np.ndarray) -> Dict[str, float]:
    """Check for fat tails (excess kurtosis of returns).

    Financial returns typically have kurtosis > 3 (leptokurtic).

    Args:
        returns: Array of log returns, shape (N,) or (N, features).

    Returns:
        Dict with kurtosis and excess_kurtosis per feature.
    """
    if returns.ndim == 1:
        returns = returns[:, None]

    results = {}
    for i in range(returns.shape[1]):
        r = returns[:, i]
        kurt = float(stats.kurtosis(r, fisher=False))  # Pearson kurtosis
        excess_kurt = float(stats.kurtosis(r, fisher=True))  # Excess kurtosis
        results[f"feature_{i}"] = {
            "kurtosis": kurt,
            "excess_kurtosis": excess_kurt,
            "is_fat_tailed": excess_kurt > 1.0,
        }
    return results


def check_volatility_clustering(
    returns: np.ndarray,
    max_lag: int = 20,
) -> Dict[str, float]:
    """Check for volatility clustering (ACF of squared/absolute returns).

    Volatility clustering: ACF of |r_t| or r_t^2 is significantly positive.

    Args:
        returns: Log returns, shape (N,).
        max_lag: Maximum lag for ACF computation.

    Returns:
        Dict with mean_abs_acf and mean_sq_acf at various lags.
    """
    abs_returns = np.abs(returns)
    sq_returns = returns ** 2

    def acf_mean(x: np.ndarray, max_lag: int) -> float:
        n = len(x)
        x_c = x - x.mean()
        var = np.var(x)
        acf_vals = []
        for lag in range(1, max_lag + 1):
            cov = np.dot(x_c[lag:], x_c[:-lag]) / (n - lag)
            acf_vals.append(abs(cov / (var + 1e-10)))
        return float(np.mean(acf_vals))

    return {
        "mean_abs_return_acf": acf_mean(abs_returns, max_lag),
        "mean_sq_return_acf": acf_mean(sq_returns, max_lag),
        "has_clustering": acf_mean(abs_returns, max_lag) > 0.05,
    }


def check_leverage_effect(
    returns: np.ndarray,
    volatility: np.ndarray,
    lags: int = 10,
) -> Dict[str, float]:
    """Check for leverage effect: negative correlation between returns and future vol.

    Args:
        returns: Log returns, shape (N,).
        volatility: Realized/rolling volatility, shape (N,).
        lags: Number of forward lags to check.

    Returns:
        Dict with correlation at each lag.
    """
    results = {}
    n = len(returns)
    for lag in range(1, lags + 1):
        if lag < n:
            corr = float(np.corrcoef(returns[:-lag], volatility[lag:])[0, 1])
            results[f"lag_{lag}"] = corr

    neg_corrs = [v for v in results.values() if not np.isnan(v)]
    results["mean_leverage_corr"] = float(np.mean(neg_corrs)) if neg_corrs else 0.0
    results["has_leverage_effect"] = results["mean_leverage_corr"] < -0.05
    return results


def check_volume_volatility_correlation(
    volume: np.ndarray,
    volatility: np.ndarray,
) -> Dict[str, float]:
    """Check for positive volume-volatility correlation.

    Args:
        volume: Trading volume, shape (N,).
        volatility: Realized volatility, shape (N,).

    Returns:
        Dict with Pearson and Spearman correlations.
    """
    pearson_corr, pearson_p = stats.pearsonr(volume, volatility)
    spearman_corr, spearman_p = stats.spearmanr(volume, volatility)
    return {
        "pearson_correlation": float(pearson_corr),
        "pearson_p_value": float(pearson_p),
        "spearman_correlation": float(spearman_corr),
        "spearman_p_value": float(spearman_p),
        "has_vol_volume_corr": pearson_corr > 0.1,
    }


def check_all_stylized_facts(
    returns: np.ndarray,
    volume: np.ndarray,
    volatility: np.ndarray,
) -> Dict[str, dict]:
    """Run all stylized facts checks.

    Args:
        returns: Log returns array, shape (N,).
        volume: Volume array, shape (N,).
        volatility: Realized volatility array, shape (N,).

    Returns:
        Nested dict with results for each stylized fact.
    """
    return {
        "fat_tails": check_fat_tails(returns),
        "volatility_clustering": check_volatility_clustering(returns),
        "leverage_effect": check_leverage_effect(returns, volatility),
        "volume_volatility_correlation": check_volume_volatility_correlation(volume, volatility),
    }
