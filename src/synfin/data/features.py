"""Feature engineering: technical indicators and market features."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import the 'ta' library for technical analysis
try:
    import ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    logger.warning(
        "Package 'ta' not installed. Technical indicators (RSI, MACD, BB, ATR) "
        "will not be available. Install with: pip install ta"
    )


def add_rsi(df: pd.DataFrame, window: int = 14, col: str = "Close") -> pd.DataFrame:
    """Add Relative Strength Index (RSI) feature.

    Args:
        df: OHLCV DataFrame.
        window: RSI window period.
        col: Price column to compute RSI on.

    Returns:
        DataFrame with added "RSI" column.
    """
    if not _HAS_TA:
        logger.warning("RSI requires the 'ta' package. Skipping.")
        return df
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(close=df[col], window=window).rsi()
    return df


def add_macd(
    df: pd.DataFrame,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    col: str = "Close",
) -> pd.DataFrame:
    """Add MACD indicator features.

    Args:
        df: OHLCV DataFrame.
        window_slow: Slow EMA window.
        window_fast: Fast EMA window.
        window_sign: Signal line window.
        col: Price column to compute MACD on.

    Returns:
        DataFrame with added "MACD", "MACD_Signal", and "MACD_Hist" columns.
    """
    if not _HAS_TA:
        logger.warning("MACD requires the 'ta' package. Skipping.")
        return df
    df = df.copy()
    macd = ta.trend.MACD(
        close=df[col],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    window_dev: float = 2.0,
    col: str = "Close",
) -> pd.DataFrame:
    """Add Bollinger Bands features.

    Args:
        df: OHLCV DataFrame.
        window: Moving average window.
        window_dev: Number of standard deviations for bands.
        col: Price column.

    Returns:
        DataFrame with added "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width" columns.
    """
    if not _HAS_TA:
        logger.warning("Bollinger Bands require the 'ta' package. Skipping.")
        return df
    df = df.copy()
    bb = ta.volatility.BollingerBands(close=df[col], window=window, window_dev=window_dev)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Width"] = bb.bollinger_wband()
    return df


def add_atr(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Add Average True Range (ATR) feature.

    Args:
        df: OHLCV DataFrame with High, Low, Close columns.
        window: ATR window period.

    Returns:
        DataFrame with added "ATR" column.
    """
    if not _HAS_TA:
        logger.warning("ATR requires the 'ta' package. Skipping.")
        return df
    df = df.copy()
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=window
    ).average_true_range()
    return df


def add_realized_volatility(
    df: pd.DataFrame,
    window: int = 21,
    returns_col: str = "LogReturn",
) -> pd.DataFrame:
    """Add rolling realized volatility (annualized std of log returns).

    Args:
        df: DataFrame containing log returns.
        window: Rolling window size.
        returns_col: Column name for log returns.

    Returns:
        DataFrame with added "RealizedVol" column.
    """
    if returns_col not in df.columns:
        logger.warning("Column %r not found. Skipping realized volatility.", returns_col)
        return df
    df = df.copy()
    df["RealizedVol"] = df[returns_col].rolling(window=window).std() * np.sqrt(252)
    return df


def add_features(
    df: pd.DataFrame,
    use_rsi: bool = True,
    use_macd: bool = True,
    use_bollinger: bool = True,
    use_atr: bool = True,
    use_realized_vol: bool = True,
    rsi_window: int = 14,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Apply all optional feature engineering steps.

    Args:
        df: OHLCV DataFrame (should already have LogReturn column for volatility).
        use_rsi: Whether to add RSI.
        use_macd: Whether to add MACD.
        use_bollinger: Whether to add Bollinger Bands.
        use_atr: Whether to add ATR.
        use_realized_vol: Whether to add realized volatility.
        rsi_window: Window size for RSI.
        vol_window: Window size for realized volatility.

    Returns:
        DataFrame with all requested features added.
    """
    if use_rsi:
        df = add_rsi(df, window=rsi_window)
    if use_macd:
        df = add_macd(df)
    if use_bollinger:
        df = add_bollinger_bands(df)
    if use_atr:
        df = add_atr(df)
    if use_realized_vol:
        df = add_realized_volatility(df, window=vol_window)
    return df
