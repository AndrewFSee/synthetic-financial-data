"""Data preprocessing: normalization, log-transforms, windowing, and splits."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Compute log returns from a price series.

    Args:
        df: DataFrame containing price data.
        price_col: Column name of the price to use.

    Returns:
        DataFrame with added "LogReturn" column.
    """
    df = df.copy()
    df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))
    return df.dropna()


def compute_log_volume(df: pd.DataFrame, volume_col: str = "Volume") -> pd.DataFrame:
    """Compute log-transformed volume.

    Args:
        df: DataFrame containing volume data.
        volume_col: Column name for volume.

    Returns:
        DataFrame with added "LogVolume" column.
    """
    df = df.copy()
    df["LogVolume"] = np.log1p(df[volume_col])
    return df


def compute_dollar_volume(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """Compute dollar volume (price × volume).

    Args:
        df: DataFrame containing price and volume data.
        price_col: Column name for price.
        volume_col: Column name for volume.

    Returns:
        DataFrame with added "DollarVolume" column.
    """
    df = df.copy()
    df["DollarVolume"] = df[price_col] * df[volume_col]
    return df


def normalize(
    df: pd.DataFrame,
    method: str = "minmax",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    """Normalize / standardize features.

    Args:
        df: DataFrame with features to normalize.
        method: Normalization method ("minmax" or "zscore").
        feature_cols: Columns to normalize. If None, normalizes all numeric columns.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler).
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "minmax":
        scaler: Union[MinMaxScaler, StandardScaler] = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. Use 'minmax' or 'zscore'.")

    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def create_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    feature_cols: Optional[List[str]] = None,
    stride: int = 1,
) -> np.ndarray:
    """Create sliding windows from a time-series DataFrame.

    Args:
        df: DataFrame with time-series data (rows = timesteps).
        window_size: Length of each window (sequence length).
        feature_cols: Columns to include in the windows. If None, uses all numeric.
        stride: Step size between windows.

    Returns:
        NumPy array of shape (num_windows, window_size, num_features).
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[feature_cols].values
    n_steps = data.shape[0]

    windows = []
    for start in range(0, n_steps - window_size + 1, stride):
        windows.append(data[start : start + window_size])

    if not windows:
        raise ValueError(
            f"DataFrame too short ({n_steps} rows) for window_size={window_size}."
        )

    return np.array(windows, dtype=np.float32)


def train_val_test_split(
    windows: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split windows into train / validation / test sets (time-based, no leakage).

    Args:
        windows: Array of shape (num_windows, window_size, num_features).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.

    Returns:
        Tuple of (train_windows, val_windows, test_windows).
    """
    n = len(windows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


def preprocess(
    df: pd.DataFrame,
    window_size: int = 30,
    normalization: str = "minmax",
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    stride: int = 1,
) -> Dict:
    """Full preprocessing pipeline: log-transforms → features → normalize → window → split.

    Args:
        df: Raw OHLCV DataFrame.
        window_size: Length of each sliding window.
        normalization: Normalization method ("minmax" or "zscore").
        feature_cols: Features to include. Defaults to [Open, High, Low, Close, Volume].
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        stride: Window stride.

    Returns:
        Dictionary with keys: train, val, test, scaler, feature_cols.
    """
    # Compute derived features
    df = compute_log_returns(df)
    df = compute_log_volume(df)
    df = compute_dollar_volume(df)

    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Close", "Volume",
                        "LogReturn", "LogVolume", "DollarVolume"]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols)

    # Normalize
    df_norm, scaler = normalize(df, method=normalization, feature_cols=feature_cols)

    # Create windows
    windows = create_windows(df_norm, window_size=window_size,
                             feature_cols=feature_cols, stride=stride)

    # Split
    train, val, test = train_val_test_split(windows, train_ratio=train_ratio,
                                            val_ratio=val_ratio)

    return {
        "train": train,
        "val": val,
        "test": test,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
