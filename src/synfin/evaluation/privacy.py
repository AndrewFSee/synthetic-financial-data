"""Privacy metrics: memorization checks against a real holdout set.

Distances to the training data only mean something relative to a reference:
how close does *unseen real data* get to the training set? Both metrics below
compare synthetic samples against such a holdout (real data the generator
never trained on). If no holdout is supplied, the last part of ``real`` is
held out chronologically, with a gap so that overlapping windows cannot leak.

Copying vs. collapse: a generator that collapses toward the dense centre of
the data also produces samples that are close to training records, without
copying any of them. Raw distance-to-closest-record (DCR) cannot tell the two
apart. :func:`membership_inference_risk` therefore uses the ratio d1/d2 of the
distances to the nearest and second-nearest training records: a copy sits on
one particular record (ratio near 0), while a collapsed sample sits between
many (ratio near 1). Collapse itself is reported by :func:`collapse_diagnostics`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1) if x.ndim > 2 else x


def _reference_split(
    real: np.ndarray, holdout: Optional[np.ndarray], holdout_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (reference, holdout); carve a chronological holdout if none given."""
    if holdout is not None:
        return _flat(real), _flat(holdout)
    gap = real.shape[1] if real.ndim > 2 else 0  # window length: no shared rows
    n_hold = int(len(real) * holdout_ratio)
    cut = len(real) - n_hold
    if cut - gap < 2 or n_hold < 2:
        raise ValueError("Not enough real samples to carve out a holdout set.")
    return _flat(real[: cut - gap]), _flat(real[cut:])


def nearest_neighbor_distance_ratio(
    real: np.ndarray,
    synthetic: np.ndarray,
    holdout: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Nearest-Neighbor Distance Ratio (NNDR) against a real holdout.

    NNDR = median nearest-neighbor distance (synthetic -> reference) /
           median nearest-neighbor distance (holdout -> reference).

    ~1.0 means synthetic samples sit as far from the training records as unseen
    real data does; values well below 1 indicate memorization. Only the single
    nearest record is used: averaging over several neighbors dilutes the signal
    from a sample that copies one particular record.

    Args:
        real: Real (training/reference) data, shape (N, ...).
        synthetic: Synthetic data, shape (M, ...).
        holdout: Unseen real data; carved from ``real`` if None.

    Returns:
        Dict with ``nndr`` and the two median distances.
    """
    reference, held = _reference_split(real, holdout)
    nn = NearestNeighbors(n_neighbors=1).fit(reference)
    med_s = float(np.median(nn.kneighbors(_flat(synthetic))[0]))
    med_h = float(np.median(nn.kneighbors(held)[0]))
    nndr = med_s / (med_h + 1e-10)
    return {
        "nndr": nndr,
        "median_dist_synthetic_to_real": med_s,
        "median_dist_holdout_to_real": med_h,
        "privacy_risk": bool(nndr < 0.5),
    }


def distance_to_closest_record(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, float]:
    """Distance to Closest Record (DCR) from each synthetic sample to ``real``.

    Args:
        real: Real data, shape (N, D).
        synthetic: Synthetic data, shape (M, D).

    Returns:
        Dict with DCR statistics.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(_flat(real))
    distances, _ = nn.kneighbors(_flat(synthetic))
    dcr = distances[:, 0]
    return {
        "mean_dcr": float(dcr.mean()),
        "median_dcr": float(np.median(dcr)),
        "min_dcr": float(dcr.min()),
        "pct_5th": float(np.percentile(dcr, 5)),
    }


def _nearest_two(reference: np.ndarray, x: np.ndarray):
    """Nearest-record distance, d1/d2 ratio and nearest-record index for each row of x."""
    nn = NearestNeighbors(n_neighbors=2).fit(reference)
    dist, idx = nn.kneighbors(x)
    return dist[:, 0], dist[:, 0] / (dist[:, 1] + 1e-12), idx[:, 0]


def membership_inference_risk(
    real: np.ndarray,
    synthetic: np.ndarray,
    threshold_percentile: float = 5.0,
    holdout: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Holdout-calibrated memorization rate based on the d1/d2 distance ratio.

    For each sample, d1/d2 is its distance to the nearest training record
    divided by the distance to the second nearest. The threshold is the
    ``threshold_percentile`` of the holdout's ratios. Without memorization,
    about that share of synthetic samples falls below it too; copies of
    training records fall far below it. Unlike raw DCR, the ratio is not
    fooled by mode collapse, and being scale-free it shifts little when the
    holdout period is calmer or more volatile than the training period.

    Args:
        real: Real (training/reference) data.
        synthetic: Synthetic data.
        threshold_percentile: Percentile of holdout ratios used as threshold.
        holdout: Unseen real data; carved from ``real`` if None.

    Returns:
        Dict with ``memorization_rate``, ``expected_rate``, their ratio
        ``memorization_lift``, the ratio threshold, and DCR statistics.
    """
    reference, held = _reference_split(real, holdout)
    dcr_s, ratio_s, _ = _nearest_two(reference, _flat(synthetic))
    dcr_h, ratio_h, _ = _nearest_two(reference, held)
    threshold = float(np.percentile(ratio_h, threshold_percentile))
    rate = float((ratio_s <= threshold).mean())
    expected = threshold_percentile / 100.0
    return {
        "memorization_rate": rate,
        "expected_rate": expected,
        "memorization_lift": rate / expected,
        "ratio_threshold": threshold,
        "median_nn_ratio": float(np.median(ratio_s)),
        "holdout_median_nn_ratio": float(np.median(ratio_h)),
        "mean_dcr": float(dcr_s.mean()),
        "median_dcr": float(np.median(dcr_s)),
        "min_dcr": float(dcr_s.min()),
        "holdout_median_dcr": float(np.median(dcr_h)),
    }


def collapse_diagnostics(
    real: np.ndarray,
    synthetic: np.ndarray,
    holdout: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Detect mode collapse: too little spread, or too few distinct neighbours.

    * ``dispersion_ratio``: median over features of std(synthetic) / std(real).
    * ``coverage``: distinct nearest training records / number of samples, for
      the synthetic data and for an equal-sized holdout sample (coverage
      depends on sample size, hence the size matching).
    * ``coverage_ratio``: synthetic coverage / holdout coverage.

    ``collapsed`` is True when either ratio is below 0.5.

    Args:
        real: Real (training/reference) data, shape (N, seq_len, F) or (N, D).
        synthetic: Synthetic data in the same layout.
        holdout: Unseen real data; carved from ``real`` if None.
        seed: Seed for the size-matching subsample.

    Returns:
        Dict with the diagnostics above.
    """
    n_feat = real.shape[-1]
    sd_r = real.reshape(-1, n_feat).std(axis=0) + 1e-12
    sd_s = synthetic.reshape(-1, n_feat).std(axis=0)
    dispersion = float(np.median(sd_s / sd_r))

    reference, held = _reference_split(real, holdout)
    syn = _flat(synthetic)
    rng = np.random.default_rng(seed)
    m = min(len(syn), len(held))
    syn = syn[rng.choice(len(syn), m, replace=False)]
    held = held[rng.choice(len(held), m, replace=False)]
    _, _, idx_s = _nearest_two(reference, syn)
    _, _, idx_h = _nearest_two(reference, held)
    cov_s = len(np.unique(idx_s)) / m
    cov_h = len(np.unique(idx_h)) / m
    coverage_ratio = cov_s / max(cov_h, 1e-12)
    return {
        "dispersion_ratio": dispersion,
        "coverage": cov_s,
        "holdout_coverage": cov_h,
        "coverage_ratio": coverage_ratio,
        "collapsed": bool(dispersion < 0.5 or coverage_ratio < 0.5),
    }
