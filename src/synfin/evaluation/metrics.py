"""Aggregate all evaluation metrics into a summary report.

Pass windows in their original (unscaled) feature units together with the
feature names. Columns are located by name (``LogReturn`` for returns;
``LogVolumeRel``, ``LogVolume`` or ``Volume`` for volume), stylized facts are
computed on the raw returns, and distance-based metrics (MMD, privacy) use
features standardized with the real data's mean/std so no single feature's
units dominate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from synfin.evaluation.discriminative import discriminative_score
from synfin.evaluation.privacy import (
    collapse_diagnostics,
    membership_inference_risk,
    nearest_neighbor_distance_ratio,
)
from synfin.evaluation.statistical_tests import (
    acf_comparison,
    cross_correlation_comparison,
    ks_test,
    mmd_rbf,
)
from synfin.evaluation.stylized_facts import check_all_stylized_facts, extreme_move_check
from synfin.evaluation.tstr import tstr_benchmark

logger = logging.getLogger(__name__)

RETURN_COLUMNS = ("LogReturn",)
VOLUME_COLUMNS = ("LogVolumeRel", "LogVolume", "Volume")


def _find(names: Optional[List[str]], candidates) -> Optional[int]:
    if not names:
        return None
    for c in candidates:
        if c in names:
            return names.index(c)
    return None


def compute_all_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    run_tstr: bool = True,
    real_holdout: Optional[np.ndarray] = None,
    ks_alpha: float = 0.05,
    mmd_bandwidth: Optional[float] = None,
    tstr_task: str = "volatility",
    seed: int = 0,
) -> Dict:
    """Compute and aggregate all evaluation metrics.

    Args:
        real: Real (training) windows, shape (N, seq_len, F), unscaled.
        synthetic: Synthetic windows, shape (M, seq_len, F), same units.
        feature_names: Feature names for the last axis. Needed for stylized
            facts and TSTR (which must know the return column).
        output_dir: If provided, save ``evaluation_report.json`` here.
        run_tstr: Whether to run the TSTR benchmark.
        real_holdout: Real windows the generator never saw (e.g. the test
            split). Used by TSTR and the privacy metrics; carved out of
            ``real`` chronologically if omitted.
        ks_alpha: Significance level for KS ``reject`` flags.
        mmd_bandwidth: RBF bandwidth for MMD (None = median heuristic).
        tstr_task: "volatility" or "direction".
        seed: Seed for subsampling.

    Returns:
        Nested dictionary with all evaluation results and an overall score.
    """
    logger.info("Computing evaluation metrics...")
    rng = np.random.default_rng(seed)
    n_features = real.shape[-1]
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(f"{len(feature_names)} feature names for {n_features} features.")

    real_2d = real.reshape(-1, n_features)
    synth_2d = synthetic.reshape(-1, n_features)

    # Standardize with real statistics for distance-based metrics.
    mu = real_2d.mean(axis=0)
    sd = real_2d.std(axis=0) + 1e-12

    def std(x: np.ndarray) -> np.ndarray:
        return (x - mu) / sd

    report: Dict = {"n_real": int(len(real)), "n_synthetic": int(len(synthetic))}

    # --- KS: one random timestep per window keeps rows ~independent ---
    logger.info("Running KS tests...")
    real_ks = real[np.arange(len(real)), rng.integers(0, real.shape[1], len(real))]
    synth_ks = synthetic[
        np.arange(len(synthetic)), rng.integers(0, synthetic.shape[1], len(synthetic))
    ]
    report["ks_tests"] = ks_test(real_ks, synth_ks, feature_names, alpha=ks_alpha)

    logger.info("Computing MMD...")
    report["mmd"] = mmd_rbf(std(real), std(synthetic), bandwidth=mmd_bandwidth, seed=seed)

    logger.info("Computing ACF comparison...")
    report["acf"] = acf_comparison(real, synthetic, feature_names=feature_names)

    logger.info("Computing cross-correlation...")
    report["cross_correlation"] = cross_correlation_comparison(real_2d, synth_2d)

    # --- Stylized facts on unscaled returns, located by name ---
    ret_idx = _find(feature_names, RETURN_COLUMNS)
    vol_idx = _find(feature_names, VOLUME_COLUMNS)
    if ret_idx is None:
        logger.warning("No 'LogReturn' feature name given; skipping stylized facts and TSTR.")
    else:
        logger.info("Checking stylized facts...")
        report["stylized_facts_real"] = check_all_stylized_facts(
            real[:, :, ret_idx], real[:, :, vol_idx] if vol_idx is not None else None
        )
        report["stylized_facts_synthetic"] = check_all_stylized_facts(
            synthetic[:, :, ret_idx],
            synthetic[:, :, vol_idx] if vol_idx is not None else None,
        )
        report["extreme_moves"] = extreme_move_check(
            real[:, :, ret_idx], synthetic[:, :, ret_idx], seed=seed
        )

    # --- Discriminative score (classifier two-sample test on window dynamics) ---
    if min(len(real), len(synthetic)) >= 20:
        logger.info("Computing discriminative score...")
        report["discriminative"] = discriminative_score(real, synthetic, ret_idx, seed=seed)

    # --- Privacy (holdout-calibrated) ---
    logger.info("Computing privacy metrics...")
    holdout_std = std(real_holdout) if real_holdout is not None else None
    report["privacy"] = {
        "nndr": nearest_neighbor_distance_ratio(std(real), std(synthetic), holdout=holdout_std),
        "membership_inference": membership_inference_risk(
            std(real), std(synthetic), holdout=holdout_std
        ),
        "collapse": collapse_diagnostics(std(real), std(synthetic), holdout=holdout_std),
    }
    mi = report["privacy"]["membership_inference"]
    if mi["memorization_rate"] > 2.0 * mi["expected_rate"]:
        report["privacy"]["verdict"] = "memorization"
    elif report["privacy"]["collapse"]["collapsed"]:
        report["privacy"]["verdict"] = "collapse"
    else:
        report["privacy"]["verdict"] = "ok"

    # --- TSTR ---
    if run_tstr and ret_idx is not None and len(real) > 50 and len(synthetic) > 50:
        logger.info("Running TSTR benchmark...")
        report["tstr"] = tstr_benchmark(
            real, synthetic, return_idx=ret_idx, task=tstr_task, real_test=real_holdout
        )

    report["realism_components"] = _realism_components(report)
    comps = report["realism_components"]
    report["realism_score"] = float(np.mean(list(comps.values()))) if comps else 0.0
    logger.info("Overall realism score: %.3f", report["realism_score"])

    if output_dir:
        _save_report(report, output_dir)
    return report


def _realism_components(report: Dict) -> Dict[str, float]:
    """Sub-scores in [0, 1], higher = more realistic.

    * ``ks``: 1 - mean KS statistic (average marginal CDF agreement).
    * ``mmd``: 1 - sqrt(MMD²), clipped to [0, 1] (joint window distribution).
    * ``tstr``: 1 - 2 * |TRTR - TSTR| AUC gap, clipped (downstream usefulness).
      0 if a classifier could not even be trained on the synthetic data (its
      labels were all one class, e.g. near-constant generated volatility).
    * ``privacy``: 1 - excess memorization rate (d1/d2 test), scaled so the
      no-memorization rate scores 1 and copying every sample scores 0. Mode
      collapse is deliberately not penalized here; the discriminative score
      already catches it, and it is reported under ``privacy.collapse``.
    * ``discriminative``: 1 - 2 * (AUC - 0.5) of a real-vs-synthetic classifier
      on per-window dynamics features (see :mod:`synfin.evaluation.discriminative`).
      The other components barely see temporal dynamics (KS is marginal-only
      and MMD on flattened windows is dominated by per-step noise), so without
      this term i.i.d. noise with the right mean/std scores almost as well as
      real data.
    """
    comps: Dict[str, float] = {}
    if report.get("ks_tests"):
        comps["ks"] = 1.0 - float(np.mean([v["statistic"] for v in report["ks_tests"].values()]))
    if "mmd" in report:
        comps["mmd"] = float(np.clip(1.0 - np.sqrt(max(report["mmd"], 0.0)), 0.0, 1.0))
    tstr = report.get("tstr", {})
    if "tstr_gap" in tstr:
        comps["tstr"] = max(0.0, 1.0 - 2.0 * abs(tstr["tstr_gap"]["auc"]))
    elif "auc" in tstr.get("trtr", {}) and "skipped" in tstr.get("tstr", {}):
        comps["tstr"] = 0.0
    if "membership_inference" in report.get("privacy", {}):
        mi = report["privacy"]["membership_inference"]
        excess = (mi["memorization_rate"] - mi["expected_rate"]) / (1.0 - mi["expected_rate"])
        comps["privacy"] = float(np.clip(1.0 - excess, 0.0, 1.0))
    if "discriminative" in report:
        comps["discriminative"] = report["discriminative"]["score"]
    return comps


def _save_report(report: Dict, output_dir: str) -> None:
    """Save the evaluation report as JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    report_path = path / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, default=_serialize, indent=2)
    logger.info("Evaluation report saved to %s", report_path)
