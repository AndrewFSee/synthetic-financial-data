"""Aggregate all evaluation metrics into a summary report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from synfin.evaluation.privacy import membership_inference_risk, nearest_neighbor_distance_ratio
from synfin.evaluation.statistical_tests import (
    acf_comparison,
    cross_correlation_comparison,
    ks_test,
    mmd_rbf,
)
from synfin.evaluation.stylized_facts import check_all_stylized_facts
from synfin.evaluation.tstr import tstr_benchmark

logger = logging.getLogger(__name__)


def compute_all_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[list] = None,
    output_dir: Optional[str] = None,
    run_tstr: bool = True,
) -> Dict:
    """Compute and aggregate all evaluation metrics.

    Args:
        real: Real data windows, shape (N, seq_len, num_features).
        synthetic: Synthetic data windows, shape (M, seq_len, num_features).
        feature_names: Optional feature names for reporting.
        output_dir: If provided, save report to this directory.
        run_tstr: Whether to run the TSTR benchmark.

    Returns:
        Nested dictionary with all evaluation results and an overall score.
    """
    logger.info("Computing evaluation metrics...")

    # Flatten windows for non-sequential tests
    real_flat = real.reshape(len(real), -1)
    synth_flat = synthetic.reshape(len(synthetic), -1)

    # Per-feature flat arrays (last feature only, for stylized facts)
    real_2d = real.reshape(-1, real.shape[-1])
    synth_2d = synthetic.reshape(-1, synthetic.shape[-1])

    report: Dict = {}

    # --- Statistical tests ---
    logger.info("Running KS tests...")
    report["ks_tests"] = ks_test(real_2d, synth_2d, feature_names)

    logger.info("Computing MMD...")
    report["mmd"] = mmd_rbf(real_flat, synth_flat)

    logger.info("Computing ACF comparison...")
    report["acf"] = acf_comparison(real_2d, synth_2d, feature_names=feature_names)

    logger.info("Computing cross-correlation...")
    report["cross_correlation"] = cross_correlation_comparison(real_2d, synth_2d)
    # Convert numpy arrays to lists for serialization
    for k in ["real_corr", "synthetic_corr", "diff"]:
        report["cross_correlation"][k] = report["cross_correlation"][k].tolist()

    # --- Stylized facts (use 5th feature = LogReturn if available) ---
    n_features = real.shape[-1]
    ret_idx = min(5, n_features - 1)
    vol_idx = min(6, n_features - 1)
    vol_col = min(4, n_features - 1)

    real_returns = real_2d[:, ret_idx]
    synth_returns = synth_2d[:, ret_idx]
    real_volume = real_2d[:, vol_col]
    real_vol = np.abs(real_returns)

    logger.info("Checking stylized facts...")
    report["stylized_facts_real"] = check_all_stylized_facts(
        real_returns, real_volume, real_vol
    )
    report["stylized_facts_synthetic"] = check_all_stylized_facts(
        synth_returns, real_volume, real_vol
    )

    # --- Privacy ---
    logger.info("Computing privacy metrics...")
    report["privacy"] = {
        "nndr": nearest_neighbor_distance_ratio(real_flat, synth_flat),
        "membership_inference": membership_inference_risk(real_flat, synth_flat),
    }

    # --- TSTR ---
    if run_tstr and len(real) > 50 and len(synthetic) > 50:
        logger.info("Running TSTR benchmark...")
        report["tstr"] = tstr_benchmark(real, synthetic)

    # --- Overall realism score ---
    report["realism_score"] = _compute_realism_score(report)
    logger.info("Overall realism score: %.3f", report["realism_score"])

    # --- Save report ---
    if output_dir:
        _save_report(report, output_dir)

    return report


def _compute_realism_score(report: Dict) -> float:
    """Compute an overall realism score from sub-metrics (0 to 1).

    Higher is better (more realistic).

    Args:
        report: The evaluation report dictionary.

    Returns:
        Scalar score in [0, 1].
    """
    scores = []

    # KS test: fraction of features with p_value > 0.05 (fail to reject H0)
    if "ks_tests" in report:
        ks_pass = [
            1.0 if v["p_value"] > 0.05 else 0.0
            for v in report["ks_tests"].values()
        ]
        if ks_pass:
            scores.append(np.mean(ks_pass))

    # MMD: lower is better; normalize as exp(-mmd)
    if "mmd" in report:
        mmd_val = report["mmd"]
        scores.append(float(np.exp(-mmd_val)))

    # TSTR: penalize gap between TRTR and TSTR
    if "tstr" in report and "tstr_gap" in report["tstr"]:
        gap = abs(report["tstr"]["tstr_gap"].get("accuracy", 0.0))
        scores.append(max(0.0, 1.0 - 2 * gap))

    # NNDR: closer to 1 is better (no memorization)
    if "privacy" in report and "nndr" in report["privacy"]:
        nndr = report["privacy"]["nndr"].get("nndr", 1.0)
        scores.append(min(1.0, nndr))

    return float(np.mean(scores)) if scores else 0.0


def _save_report(report: Dict, output_dir: str) -> None:
    """Save the evaluation report as JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, bool):
            return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    report_path = path / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, default=_serialize, indent=2)
    logger.info("Evaluation report saved to %s", report_path)
