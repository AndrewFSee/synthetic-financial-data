"""Privacy metrics: nearest-neighbor distance and membership inference."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def nearest_neighbor_distance_ratio(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_neighbors: int = 5,
) -> Dict[str, float]:
    """Compute Nearest-Neighbor Distance Ratio (NNDR).

    NNDR compares:
      - Distance from each synthetic sample to its nearest real sample
      - Distance from each real sample to its nearest real sample (leave-one-out)

    A ratio close to 1.0 indicates synthetic data is as far from real data as
    real data is from itself (good privacy). A ratio near 0 indicates memorization.

    Args:
        real: Real data, shape (N, D). Flattened if > 2D.
        synthetic: Synthetic data, shape (M, D).
        n_neighbors: Number of neighbors to consider.

    Returns:
        Dict with mean NNDR and related statistics.
    """
    if real.ndim > 2:
        real = real.reshape(len(real), -1)
    if synthetic.ndim > 2:
        synthetic = synthetic.reshape(len(synthetic), -1)

    # Distances from synthetic to real
    nn_sr = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_sr.fit(real)
    dist_s2r, _ = nn_sr.kneighbors(synthetic)
    mean_dist_s2r = float(dist_s2r[:, 1:].mean())

    # Distances from real to real (leave-one-out)
    nn_rr = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_rr.fit(real)
    dist_r2r, _ = nn_rr.kneighbors(real)
    mean_dist_r2r = float(dist_r2r[:, 1:].mean())  # exclude self (dist=0)

    nndr = mean_dist_s2r / (mean_dist_r2r + 1e-10)

    return {
        "nndr": nndr,
        "mean_dist_synthetic_to_real": mean_dist_s2r,
        "mean_dist_real_to_real": mean_dist_r2r,
        "privacy_risk": nndr < 0.5,  # Low NNDR = high memorization risk
    }


def distance_to_closest_record(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, float]:
    """Compute Distance to Closest Record (DCR).

    Args:
        real: Real data, shape (N, D).
        synthetic: Synthetic data, shape (M, D).

    Returns:
        Dict with DCR statistics.
    """
    if real.ndim > 2:
        real = real.reshape(len(real), -1)
    if synthetic.ndim > 2:
        synthetic = synthetic.reshape(len(synthetic), -1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(real)
    distances, _ = nn.kneighbors(synthetic)
    dcr = distances[:, 0]

    return {
        "mean_dcr": float(dcr.mean()),
        "median_dcr": float(np.median(dcr)),
        "min_dcr": float(dcr.min()),
        "pct_5th": float(np.percentile(dcr, 5)),
    }


def membership_inference_risk(
    real: np.ndarray,
    synthetic: np.ndarray,
    threshold_percentile: float = 5.0,
) -> Dict[str, float]:
    """Estimate membership inference attack risk.

    Approximates the fraction of synthetic samples that are suspiciously
    close to real training samples (potential memorization).

    Args:
        real: Real (training) data.
        synthetic: Synthetic data.
        threshold_percentile: DCR percentile below which samples are "at risk".

    Returns:
        Dict with risk metrics.
    """
    dcr_result = distance_to_closest_record(real, synthetic)
    nn = NearestNeighbors(n_neighbors=1)
    if real.ndim > 2:
        real = real.reshape(len(real), -1)
    if synthetic.ndim > 2:
        synthetic = synthetic.reshape(len(synthetic), -1)

    nn.fit(real)
    distances, _ = nn.kneighbors(synthetic)
    dcr = distances[:, 0]

    threshold = np.percentile(dcr, threshold_percentile)
    at_risk = float((dcr <= threshold).mean())

    return {
        **dcr_result,
        "membership_inference_risk": at_risk,
        "threshold": float(threshold),
    }
