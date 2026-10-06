"""Noise-aware agreement score for robust stylized facts.

Each stylized fact is compared through a z-score that accounts for its own
sampling noise::

    z = (synthetic - real) / sqrt(se_real^2 + se_synthetic^2)
    term = max(0, 1 - |z| / 4)      # within noise ~0.75-1, 2 SE -> 0.5, >= 4 SE -> 0

Standard errors come from bootstraps: stride-1 sliding windows (detected
automatically, normally the real data) are resampled in contiguous blocks,
because neighbouring windows share most of their days; independent windows
(normally generator output) are resampled individually. Fixed tolerances don't
work here: independent replicates of the *same* process at AAPL-like sample
sizes scatter by 10-50% on several of these statistics.

The size-matched extreme-move check is reported elsewhere but not scored: its
reference is a single real maximum, which is too noisy to score against.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats

from synfin.evaluation.statistical_tests import pooled_acf

STATS = (
    "tail_weight",
    "tail_quantile",
    "volatility_clustering",
    "extreme_clustering",
    "regime_dispersion",
    "volume_volatility",
)


def _extreme_clustering(a: np.ndarray, horizon: int = 3) -> float:
    hi, big = np.percentile(a, 99), np.percentile(a, 90)
    hits = n = 0
    for k in range(1, min(horizon, a.shape[1] - 1) + 1):
        src = a[:, :-k] >= hi
        hits += int((src & (a[:, k:] >= big)).sum())
        n += int(src.sum())
    return hits / n if n else 0.0


def stylized_stats(returns: np.ndarray, volume: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Robust stylized-fact statistics of return windows, shape (N, T).

    * ``tail_weight``: excess kurtosis with the top 0.1% of |r| trimmed
    * ``tail_quantile``: 99% quantile of |r - mean| in standard deviations
    * ``volatility_clustering``: pooled within-window ACF of |r|, lags 1-10
    * ``extreme_clustering``: P(top-10% move within 3 steps | top-1% move)
    * ``regime_dispersion``: std of log window volatility
    * ``volume_volatility``: Spearman corr(volume, |r|) (if volume given)
    """
    r = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    flat = r.ravel()
    dev = np.abs(flat - flat.mean()) / (flat.std() + 1e-12)
    keep = dev < np.percentile(dev, 99.9)
    a = np.abs(r)
    out = {
        "tail_weight": float(stats.kurtosis(flat[keep])),
        "tail_quantile": float(np.percentile(dev, 99)),
        "volatility_clustering": float(pooled_acf(a, 10)[1:].mean()),
        "extreme_clustering": _extreme_clustering(a),
        "regime_dispersion": float(np.log(r.std(axis=1) + 1e-12).std()),
    }
    if volume is not None:
        out["volume_volatility"] = float(
            stats.spearmanr(np.asarray(volume).ravel(), a.ravel()).statistic
        )
    return out


def _bootstrap_se(
    returns: np.ndarray,
    volume: Optional[np.ndarray],
    block: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap standard errors of :func:`stylized_stats` (block-resampling windows)."""
    n = len(returns)
    n_blocks = max(1, int(np.ceil(n / block)))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws.append(stylized_stats(returns[idx], volume[idx] if volume is not None else None))
    return {k: float(np.std([d[k] for d in draws], ddof=1)) for k in draws[0]}


def stylized_agreement(
    real_returns: np.ndarray,
    synthetic_returns: np.ndarray,
    real_volume: Optional[np.ndarray] = None,
    synthetic_volume: Optional[np.ndarray] = None,
    n_boot: int = 100,
    seed: int = 0,
) -> Dict:
    """Noise-aware agreement in [0, 1] between real and synthetic stylized facts.

    Args:
        real_returns: Real return windows (N, T), time-ordered (may overlap).
        synthetic_returns: Synthetic return windows (M, T), independent.
        real_volume: Optional real volume windows, same shape as real_returns.
        synthetic_volume: Optional synthetic volume windows.
        n_boot: Bootstrap replicates for each standard error.
        seed: Random seed.

    Returns:
        Dict with per-statistic ``real``, ``synthetic``, ``se``, ``z`` and
        ``term`` values under ``terms``, and their mean as ``score``.
    """
    rng = np.random.default_rng(seed)
    use_volume = real_volume is not None and synthetic_volume is not None
    rv = real_volume if use_volume else None
    sv = synthetic_volume if use_volume else None
    real = stylized_stats(real_returns, rv)
    synth = stylized_stats(synthetic_returns, sv)
    T = real_returns.shape[-1]

    def block_for(w: np.ndarray) -> int:
        # Overlapping stride-1 windows share most of their days: resample blocks.
        sliding = len(w) > 1 and np.allclose(w[:-1, 1:], w[1:, :-1])
        return 2 * T if sliding else 1

    se_real = _bootstrap_se(real_returns, rv, block_for(real_returns), n_boot, rng)
    se_synth = _bootstrap_se(synthetic_returns, sv, block_for(synthetic_returns), n_boot, rng)
    terms = {}
    for k in real:
        se = float(np.sqrt(se_real[k] ** 2 + se_synth[k] ** 2)) + 1e-12
        z = (synth[k] - real[k]) / se
        terms[k] = {
            "real": real[k],
            "synthetic": synth[k],
            "se": se,
            "z": float(z),
            "term": float(max(0.0, 1.0 - abs(z) / 4.0)),
        }
    return {"terms": terms, "score": float(np.mean([t["term"] for t in terms.values()]))}
