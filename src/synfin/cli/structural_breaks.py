"""Generate labeled structural-break datasets in the ADIA Lab formats: ``synfin-structural-breaks``.

Examples:
    # 10k series shaped like the 2025 offline competition
    python scripts/generate_structural_breaks.py --preset adia_offline --n-series 10000

    # Real-time layout, breaks twice as large (easier), plus a difficulty report
    python scripts/generate_structural_breaks.py --preset adia_realtime \
        --magnitude-scale 2.0 --report

    # Use real AAPL returns (block-bootstrapped) as the no-break base process
    python scripts/generate_structural_breaks.py --bootstrap-from data/raw/AAPL_1d.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from synfin.structural_breaks import (
    PRESETS,
    auc_report,
    baseline_scores,
    generate,
    get_preset,
    save_adia,
    summarize,
)
from synfin.utils.logging import setup_logging


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labeled structural-break data for testing detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default="adia_offline", choices=sorted(PRESETS))
    parser.add_argument("--n-series", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--magnitude-scale",
        type=float,
        default=1.0,
        help="Multiplier on break sizes (>1 = easier, <1 = harder)",
    )
    parser.add_argument("--break-prob", type=float, default=None, help="Override break rate")
    parser.add_argument(
        "--bootstrap-from",
        default=None,
        help="Parquet/CSV of OHLCV prices; its Close log returns become the null process",
    )
    parser.add_argument("--split", default="train", help="File-name suffix (train/test)")
    parser.add_argument(
        "--output-dir", default=None, help="Default: data/structural_breaks/<preset>"
    )
    parser.add_argument("--report", action="store_true", help="Print a baseline-AUC report")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _load_returns(path: str) -> np.ndarray:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return np.diff(np.log(df["Close"].to_numpy(dtype=float)))


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    cfg = get_preset(args.preset, magnitude_scale=args.magnitude_scale)
    if args.break_prob is not None:
        cfg.layout.break_prob = args.break_prob
    source = _load_returns(args.bootstrap_from) if args.bootstrap_from else None

    logger.info("Generating %d series (preset=%s, seed=%d)", args.n_series, args.preset, args.seed)
    ds = generate(args.n_series, cfg, seed=args.seed, bootstrap_source=source)

    layout = "offline" if cfg.layout.break_at_boundary else "realtime"
    out = args.output_dir or f"data/structural_breaks/{args.preset}"
    save_adia(ds, out, layout=layout, split=args.split)

    if args.report:
        print("\nHistory stylized facts (medians):")
        for k, v in summarize(ds).items():
            print(f"  {k:16s} {v:+.3f}")
        scores = baseline_scores(ds)
        print("\nBaseline detector AUCs (history vs online segment):")
        for col in scores:
            report = auc_report(scores[col], ds.meta)
            print(f"\n[{col}]")
            print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
