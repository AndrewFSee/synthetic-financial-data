"""Download OHLCV data: ``synfin-download``.

Defaults for tickers, dates, interval and output directory come from the
``data:`` section of ``--config`` (``configs/default.yaml``); flags override.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from synfin.data.download import download_ohlcv
from synfin.utils.config import load_config
from synfin.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV market data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tickers", nargs="+", default=None, help="Default: data.tickers")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default=None, help="Data interval (1d, 1h, ...)")
    parser.add_argument("--output-dir", default=None, help="Default: data.raw_dir")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)
    data_cfg = load_config(args.config).get("data", {}) if Path(args.config).exists() else {}

    tickers = args.tickers or data_cfg.get("tickers", ["AAPL"])
    start = args.start or data_cfg.get("start_date", "2015-01-01")
    end = args.end or data_cfg.get("end_date", "2024-12-31")
    interval = args.interval or data_cfg.get("interval", "1d")
    logger.info("Downloading %s (%s to %s, interval=%s)", tickers, start, end, interval)

    results = download_ohlcv(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        output_dir=args.output_dir or data_cfg.get("raw_dir", "data/raw"),
        save_format=args.format,
    )
    logger.info("Downloaded data for %d tickers:", len(results))
    for ticker, df in results.items():
        logger.info("  %s: %d rows, %s to %s", ticker, len(df), df.index[0], df.index[-1])


if __name__ == "__main__":
    main()
