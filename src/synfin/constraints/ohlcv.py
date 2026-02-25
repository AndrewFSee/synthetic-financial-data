"""Post-processing constraints to enforce valid OHLCV relationships."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def enforce_ohlc_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce OHLC relationships: Low ≤ min(Open, Close) and High ≥ max(Open, Close).

    Args:
        df: DataFrame with Open, High, Low, Close columns.

    Returns:
        DataFrame with corrected OHLC values.
    """
    df = df.copy()
    df["Low"] = np.minimum(df["Low"], np.minimum(df["Open"], df["Close"]))
    df["High"] = np.maximum(df["High"], np.maximum(df["Open"], df["Close"]))
    return df


def enforce_volume_constraints(
    df: pd.DataFrame,
    min_volume: float = 1.0,
    round_to_int: bool = False,
) -> pd.DataFrame:
    """Enforce Volume > 0 and optionally round to integer shares.

    Args:
        df: DataFrame with Volume column.
        min_volume: Minimum volume value (default 1.0).
        round_to_int: Whether to round Volume to nearest integer.

    Returns:
        DataFrame with corrected Volume values.
    """
    df = df.copy()
    df["Volume"] = np.maximum(df["Volume"], min_volume)
    if round_to_int:
        df["Volume"] = df["Volume"].round().astype(int)
    return df


def enforce_price_continuity(
    df: pd.DataFrame,
    max_gap_pct: float = 0.20,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Clip unrealistically large price gaps between consecutive bars.

    Args:
        df: DataFrame with price data.
        max_gap_pct: Maximum allowed percentage price change between bars.
        price_col: Price column to check for gaps.

    Returns:
        DataFrame with corrected price continuity.
    """
    df = df.copy()
    prices = df[price_col].values.copy()
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            pct_change = abs(prices[i] - prices[i - 1]) / abs(prices[i - 1])
            if pct_change > max_gap_pct:
                direction = np.sign(prices[i] - prices[i - 1])
                prices[i] = prices[i - 1] * (1 + direction * max_gap_pct)
    df[price_col] = prices
    return df


def enforce_positive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all price columns are positive.

    Args:
        df: DataFrame with OHLC price columns.

    Returns:
        DataFrame with non-positive prices replaced by a small positive value.
    """
    df = df.copy()
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    for col in price_cols:
        df[col] = np.maximum(df[col], 1e-6)
    return df


def apply_all_constraints(
    df: pd.DataFrame,
    round_volume: bool = False,
    max_gap_pct: float = 0.20,
) -> pd.DataFrame:
    """Apply all OHLCV constraints in sequence.

    Order: positive prices → OHLC relationships → volume → price continuity.

    Args:
        df: OHLCV DataFrame to post-process.
        round_volume: Whether to round volume to integer.
        max_gap_pct: Maximum allowed price gap percentage.

    Returns:
        Corrected OHLCV DataFrame.
    """
    df = enforce_positive_prices(df)
    df = enforce_ohlc_constraints(df)
    df = enforce_volume_constraints(df, round_to_int=round_volume)
    df = enforce_price_continuity(df, max_gap_pct=max_gap_pct)
    return df
