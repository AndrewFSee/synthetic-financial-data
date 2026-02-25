#!/usr/bin/env python
"""Run full evaluation suite comparing real and synthetic data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from synfin.evaluation.metrics import compute_all_metrics
from synfin.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic financial data quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-data", required=True, help="Path to real data (parquet/npy)")
    parser.add_argument("--synthetic-data", required=True, help="Path to synthetic data (parquet/npy)")
    parser.add_argument("--output", default="reports/", help="Output directory for reports")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--no-tstr", action="store_true", help="Skip TSTR benchmark")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_data(path: str, window_size: int) -> np.ndarray:
    """Load data and reshape to (N, window_size, features) if needed."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix in [".parquet", ".csv"]:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        data = df.select_dtypes(include=[np.number]).values
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if data.ndim == 2:
        # Create windows
        n = len(data) - window_size
        windows = np.stack([data[i:i + window_size] for i in range(n)])
        return windows
    return data


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Loading real data from %s", args.real_data)
    real = load_data(args.real_data, args.window_size)
    logger.info("Loading synthetic data from %s", args.synthetic_data)
    synthetic = load_data(args.synthetic_data, args.window_size)

    logger.info("Real shape: %s, Synthetic shape: %s", real.shape, synthetic.shape)

    report = compute_all_metrics(
        real=real,
        synthetic=synthetic,
        output_dir=args.output,
        run_tstr=not args.no_tstr,
    )

    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Overall Realism Score: {report['realism_score']:.3f} / 1.000")
    print(f"  MMD: {report['mmd']:.4f}")
    if "tstr" in report:
        print(f"  TSTR Accuracy Gap: {report['tstr']['tstr_gap']['accuracy']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
