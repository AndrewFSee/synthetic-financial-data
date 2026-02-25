"""Download OHLCV market data using yfinance."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_ohlcv(
    tickers: Union[str, List[str]],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
    output_dir: Union[str, Path] = "data/raw",
    save_format: str = "parquet",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for one or more tickers using yfinance.

    Args:
        tickers: Single ticker symbol or list of ticker symbols.
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.
        interval: Data interval (e.g., "1d", "1h", "1wk").
        output_dir: Directory to save downloaded data.
        save_format: File format to save data ("parquet" or "csv").

    Returns:
        Dictionary mapping ticker symbol to DataFrame with OHLCV data.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        logger.info("Downloading %s from %s to %s (interval=%s)", ticker, start, end, interval)

        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if raw.empty:
                logger.warning("No data returned for ticker %s", ticker)
                continue

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]

            # Standardize column names
            raw = raw.rename(
                columns={
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }
            )
            raw.index.name = "Date"
            raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

            # Save to disk
            filename = f"{ticker}_{interval}.{save_format}"
            filepath = output_dir / filename
            if save_format == "parquet":
                raw.to_parquet(filepath)
            else:
                raw.to_csv(filepath)

            logger.info("Saved %s rows for %s to %s", len(raw), ticker, filepath)
            results[ticker] = raw

        except Exception as exc:
            logger.error("Failed to download %s: %s", ticker, exc)

    return results


def load_ohlcv(
    ticker: str,
    interval: str = "1d",
    data_dir: Union[str, Path] = "data/raw",
    file_format: str = "parquet",
) -> Optional[pd.DataFrame]:
    """Load previously downloaded OHLCV data from disk.

    Args:
        ticker: Ticker symbol.
        interval: Data interval.
        data_dir: Directory containing raw data.
        file_format: File format ("parquet" or "csv").

    Returns:
        DataFrame with OHLCV data, or None if file not found.
    """
    data_dir = Path(data_dir)
    filename = f"{ticker}_{interval}.{file_format}"
    filepath = data_dir / filename

    if not filepath.exists():
        logger.warning("Data file not found: %s", filepath)
        return None

    if file_format == "parquet":
        return pd.read_parquet(filepath)
    return pd.read_csv(filepath, index_col="Date", parse_dates=True)


def main() -> None:
    """CLI entry point for downloading data."""
    import argparse

    parser = argparse.ArgumentParser(description="Download OHLCV market data")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_ohlcv(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        output_dir=args.output_dir,
        save_format=args.format,
    )
