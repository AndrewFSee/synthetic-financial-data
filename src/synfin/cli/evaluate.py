"""Evaluate synthetic windows against real data: ``synfin-evaluate``.

Inputs:
    --real-data       the ``<TICKER>_windows.npz`` bundle written by synfin-train
                      (its train split is the reference, its test split the holdout),
                      or a raw ``.npy`` window array
    --synthetic-data  the ``.npz`` written by synfin-generate, or a ``.npy`` array

Both sides are compared in original feature units (the bundle's windows are
un-scaled with its stored scaler).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from synfin.data.preprocess import invert_scaling
from synfin.evaluation.metrics import compute_all_metrics
from synfin.utils.config import load_config
from synfin.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic financial data quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-data", required=True, help="Window bundle (.npz) or .npy")
    parser.add_argument("--synthetic-data", required=True, help="Generated .npz or .npy")
    parser.add_argument("--config", default="configs/default.yaml", help="Uses `evaluation:`")
    parser.add_argument("--output", default=None, help="Default: evaluation.output_dir")
    parser.add_argument("--feature-names", nargs="+", default=None, help="For .npy inputs")
    parser.add_argument("--no-tstr", action="store_true", help="Skip TSTR benchmark")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_real(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """Return (reference windows, holdout windows or None, feature names or None)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p), None, None
    with np.load(p) as z:
        params = {"center": z["scaler_center"], "scale": z["scaler_scale"]}
        train = invert_scaling(z["train"], params)
        test = invert_scaling(z["test"], params) if len(z["test"]) else None
        return train, test, [str(c) for c in z["feature_cols"]]


def load_synthetic(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return (synthetic windows, feature names or None)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p), None
    with np.load(p) as z:
        return z["windows"], [str(c) for c in z["feature_names"]]


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)
    eval_cfg = load_config(args.config).get("evaluation", {}) if Path(args.config).exists() else {}

    real, holdout, real_names = load_real(args.real_data)
    synthetic, synth_names = load_synthetic(args.synthetic_data)
    if real_names and synth_names and real_names != synth_names:
        raise ValueError(f"Feature mismatch: real {real_names} vs synthetic {synth_names}")
    if real.shape[1:] != synthetic.shape[1:]:
        raise ValueError(f"Shape mismatch: real {real.shape} vs synthetic {synthetic.shape}")
    names = args.feature_names or real_names or synth_names
    logger.info(
        "Real %s (holdout %s), synthetic %s",
        real.shape,
        None if holdout is None else holdout.shape,
        synthetic.shape,
    )

    report = compute_all_metrics(
        real=real,
        synthetic=synthetic,
        feature_names=names,
        output_dir=args.output or eval_cfg.get("output_dir", "reports"),
        run_tstr=not args.no_tstr,
        real_holdout=holdout,
        ks_alpha=eval_cfg.get("ks_alpha", 0.05),
        mmd_bandwidth=eval_cfg.get("mmd_bandwidth"),
        tstr_task=eval_cfg.get("tstr_task", "volatility"),
    )

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Overall realism score: {report['realism_score']:.3f} / 1.000")
    for k, v in report["realism_components"].items():
        print(f"    {k:14s} {v:.3f}")
    print(f"  MMD^2 (median-heuristic RBF): {report['mmd']:.4f}")
    if "tstr_gap" in report.get("tstr", {}):
        t = report["tstr"]
        print(
            f"  TSTR ({t['task']['name']}): TRTR AUC {t['trtr']['auc']:.3f}  "
            f"TSTR AUC {t['tstr']['auc']:.3f}"
        )
    priv = report["privacy"]
    mi, col = priv["membership_inference"], priv["collapse"]
    print(f"  Privacy verdict: {priv['verdict']}")
    print(
        f"    memorization rate {mi['memorization_rate']:.3f} "
        f"(expected {mi['expected_rate']:.3f} without memorization)"
    )
    print(
        f"    spread vs real {col['dispersion_ratio']:.2f}, nearest-record coverage "
        f"vs holdout {col['coverage_ratio']:.2f} (either < 0.5 = collapse)"
    )
    if "stylized_facts_real" in report:
        r, s = report["stylized_facts_real"], report["stylized_facts_synthetic"]
        print("  Stylized facts (real vs synthetic):")
        print(
            f"    excess kurtosis      {r['fat_tails']['excess_kurtosis']:7.3f}  "
            f"{s['fat_tails']['excess_kurtosis']:7.3f}"
        )
        print(
            f"    |r| ACF (mean 1-10)  {r['volatility_clustering']['mean_abs_return_acf']:7.3f}  "
            f"{s['volatility_clustering']['mean_abs_return_acf']:7.3f}"
        )
        print(
            f"    leverage corr        {r['leverage_effect']['mean_leverage_corr']:7.3f}  "
            f"{s['leverage_effect']['mean_leverage_corr']:7.3f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
