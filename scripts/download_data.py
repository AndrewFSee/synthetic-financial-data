#!/usr/bin/env python
"""Download and preprocess OHLCV financial data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synfin.data.download import download_ohlcv
from synfin.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OHLCV market data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"],
        help="Ticker symbols to download",
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Data interval (1d, 1h, etc.)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument(
        "--format", default="parquet", choices=["parquet", "csv"],
        help="File format to save data",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(
        "Downloading data for tickers: %s (%s to %s, interval=%s)",
        args.tickers, args.start, args.end, args.interval,
    )

    results = download_ohlcv(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        output_dir=args.output_dir,
        save_format=args.format,
    )

    logger.info("Downloaded data for %d tickers:", len(results))
    for ticker, df in results.items():
        logger.info("  %s: %d rows, %s to %s", ticker, len(df), df.index[0], df.index[-1])


if __name__ == "__main__":
    main()
