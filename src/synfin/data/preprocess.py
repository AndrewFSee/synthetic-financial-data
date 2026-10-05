"""Data preprocessing: feature construction, leak-free scaling, windowing and splits.

Two feature sets are available:

* ``"returns"`` (default) — stationary, scale-free features that can be mapped
  back to valid OHLCV bars (see :func:`windows_to_ohlcv`):

  ============== ==========================================================
  LogReturn      log(C_t / C_{t-1})
  OpenGap        log(O_t / C_{t-1})
  HighRange      log(H_t / max(O_t, C_t))          (>= 0)
  LowRange       log(min(O_t, C_t) / L_t)          (>= 0)
  LogVolumeRel   log1p(V_t) - mean(log1p(V)) over the previous ``volume_window`` bars
  ============== ==========================================================

* ``"levels"`` — the legacy raw price/volume levels plus derived columns. Price
  levels trend over years, so models trained on them mostly learn the price
  level of the training period; prefer ``"returns"``.

Rows are split chronologically into train/val/test *before* the scaler is fit
(on train rows only) and before windowing, so no statistic or window crosses a
split boundary.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

RETURN_FEATURES = ["LogReturn", "OpenGap", "HighRange", "LowRange", "LogVolumeRel"]
LEVEL_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "LogReturn",
    "LogVolume",
    "DollarVolume",
]
FEATURE_SETS = {"returns": RETURN_FEATURES, "levels": LEVEL_FEATURES}

# Indicator flag -> scale-free column it contributes.
INDICATOR_COLUMNS = {
    "rsi": "RSI",
    "macd": "MACD_Hist_pct",
    "bollinger": "BB_Width",
    "atr": "ATR_pct",
}


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


def compute_return_features(df: pd.DataFrame, volume_window: int = 20) -> pd.DataFrame:
    """Add the stationary ``"returns"`` feature set (see module docstring).

    Args:
        df: OHLCV DataFrame.
        volume_window: Trailing window for the relative log-volume baseline.

    Returns:
        DataFrame with the RETURN_FEATURES columns added (leading NaN rows kept).
    """
    df = df.copy()
    prev_close = df["Close"].shift(1)
    body_hi = np.maximum(df["Open"], df["Close"])
    body_lo = np.minimum(df["Open"], df["Close"])
    df["LogReturn"] = np.log(df["Close"] / prev_close)
    df["OpenGap"] = np.log(df["Open"] / prev_close)
    df["HighRange"] = np.log(np.maximum(df["High"], body_hi) / body_hi)
    df["LowRange"] = np.log(body_lo / np.minimum(df["Low"], body_lo))
    log_vol = np.log1p(df["Volume"])
    df["LogVolumeRel"] = log_vol - log_vol.shift(1).rolling(volume_window).mean()
    return df


def add_indicator_features(df: pd.DataFrame, indicators: Sequence[str]) -> pd.DataFrame:
    """Add scale-free versions of technical indicators.

    Raw MACD / ATR / Bollinger bands are in price units and trend with the
    price, so they are divided by the close (or use the band width).

    Args:
        df: OHLCV DataFrame.
        indicators: Subset of ``{"rsi", "macd", "bollinger", "atr"}``.

    Returns:
        DataFrame with the corresponding INDICATOR_COLUMNS added.
    """
    from synfin.data.features import add_atr, add_bollinger_bands, add_macd, add_rsi

    unknown = set(indicators) - set(INDICATOR_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown indicators {sorted(unknown)}; use {sorted(INDICATOR_COLUMNS)}.")
    df = df.copy()
    if "rsi" in indicators:
        df = add_rsi(df)
        if "RSI" in df:
            df["RSI"] = df["RSI"] / 100.0
    if "macd" in indicators:
        df = add_macd(df)
        if "MACD_Hist" in df:
            df["MACD_Hist_pct"] = df["MACD_Hist"] / df["Close"]
    if "bollinger" in indicators:
        df = add_bollinger_bands(df)
    if "atr" in indicators:
        df = add_atr(df)
        if "ATR" in df:
            df["ATR_pct"] = df["ATR"] / df["Close"]
    return df


def normalize(
    df: pd.DataFrame,
    method: str = "minmax",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    """Normalize / standardize features (fit and transform on the same frame).

    For train/test pipelines use :func:`fit_scaler` on the training rows only.

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
    scaler = fit_scaler(df, method, feature_cols)
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def fit_scaler(
    df: pd.DataFrame, method: str, feature_cols: List[str]
) -> Union[MinMaxScaler, StandardScaler]:
    """Fit a scaler on ``df[feature_cols]``.

    Args:
        df: Rows to fit on (normally the training rows only).
        method: "minmax" or "zscore".
        feature_cols: Columns to fit.

    Returns:
        The fitted sklearn scaler.
    """
    if method == "minmax":
        scaler: Union[MinMaxScaler, StandardScaler] = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. Use 'minmax' or 'zscore'.")
    scaler.fit(df[feature_cols])
    return scaler


def scaler_params(scaler: Union[MinMaxScaler, StandardScaler]) -> Dict[str, List[float]]:
    """Serialize a fitted scaler as plain lists: ``scaled = (x - center) / scale``.

    Plain lists keep checkpoints loadable with ``torch.load(weights_only=True)``.
    """
    if isinstance(scaler, MinMaxScaler):
        center, scale = scaler.data_min_, 1.0 / scaler.scale_
    else:
        center, scale = scaler.mean_, scaler.scale_
    return {"center": [float(v) for v in center], "scale": [float(v) for v in scale]}


def apply_scaling(x: np.ndarray, params: Dict[str, List[float]]) -> np.ndarray:
    """Scale raw features with parameters from :func:`scaler_params`."""
    return (x - np.asarray(params["center"])) / np.asarray(params["scale"])


def invert_scaling(x: np.ndarray, params: Dict[str, List[float]]) -> np.ndarray:
    """Map scaled features back to their original units."""
    return x * np.asarray(params["scale"]) + np.asarray(params["center"])


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
        raise ValueError(f"DataFrame too short ({n_steps} rows) for window_size={window_size}.")

    return np.array(windows, dtype=np.float32)


def train_val_test_split(
    windows: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an ordered array into train / validation / test parts (no shuffling).

    Args:
        windows: Array whose first axis is time-ordered.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.

    Returns:
        Tuple of (train, val, test).
    """
    n = len(windows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


def _windows_or_empty(
    df: pd.DataFrame, window_size: int, feature_cols: List[str], stride: int
) -> np.ndarray:
    if len(df) < window_size:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32)
    return create_windows(df, window_size=window_size, feature_cols=feature_cols, stride=stride)


def preprocess(
    df: pd.DataFrame,
    window_size: int = 30,
    normalization: str = "zscore",
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    stride: int = 1,
    feature_set: str = "returns",
    indicators: Sequence[str] = (),
    volume_window: int = 20,
) -> Dict:
    """Full pipeline: features -> chronological row split -> scale (train-fit) -> windows.

    Args:
        df: Raw OHLCV DataFrame (time-ordered).
        window_size: Length of each sliding window.
        normalization: "zscore" or "minmax".
        feature_cols: Explicit feature columns; defaults to the ``feature_set``
            columns plus any indicator columns.
        train_ratio: Fraction of rows for training.
        val_ratio: Fraction of rows for validation (the rest is test).
        stride: Window stride.
        feature_set: "returns" (stationary, default) or "levels" (legacy).
        indicators: Optional indicators to append (see :func:`add_indicator_features`).
        volume_window: Baseline window for ``LogVolumeRel``.

    Returns:
        Dict with keys ``train``, ``val``, ``test`` (scaled windows), ``scaler``,
        ``scaler_params``, ``feature_cols``, ``feature_set`` and
        ``log_volume_base`` (mean log1p volume of the training rows, used to
        rebuild volumes from ``LogVolumeRel``).
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}. Use {sorted(FEATURE_SETS)}.")

    if feature_set == "returns":
        df = compute_return_features(df, volume_window=volume_window)
    else:
        df = compute_log_returns(df)
        df = compute_log_volume(df)
        df = compute_dollar_volume(df)
    if indicators:
        df = add_indicator_features(df, indicators)

    if feature_cols is None:
        feature_cols = list(FEATURE_SETS[feature_set])
        feature_cols += [INDICATOR_COLUMNS[i] for i in indicators]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Dropping unavailable feature columns: %s", missing)
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    train_df, val_df, test_df = train_val_test_split(df, train_ratio, val_ratio)
    scaler = fit_scaler(train_df, normalization, feature_cols)

    def scaled_windows(part: pd.DataFrame) -> np.ndarray:
        part = part.copy()
        if len(part):
            part[feature_cols] = scaler.transform(part[feature_cols])
        return _windows_or_empty(part, window_size, feature_cols, stride)

    log_volume_base = float(np.log1p(train_df["Volume"]).mean()) if "Volume" in train_df else 0.0
    return {
        "train": scaled_windows(train_df),
        "val": scaled_windows(val_df),
        "test": scaled_windows(test_df),
        "scaler": scaler,
        "scaler_params": scaler_params(scaler),
        "feature_cols": feature_cols,
        "feature_set": feature_set,
        "log_volume_base": log_volume_base,
    }


def windows_to_ohlcv(
    windows: np.ndarray,
    feature_cols: Sequence[str],
    start_close: float = 100.0,
    log_volume_base: float = 0.0,
) -> np.ndarray:
    """Rebuild OHLCV bars from unscaled ``"returns"``-set windows.

    Each window starts from ``start_close``. Range features are clipped at zero,
    so the result always satisfies Low <= min(Open, Close) <= max(Open, Close)
    <= High. Volumes use a constant baseline (``log_volume_base``) in place of
    the trailing mean, so they are approximate.

    Args:
        windows: Unscaled windows, shape (N, seq_len, F).
        feature_cols: Column names matching the last axis.
        start_close: Close price preceding each window.
        log_volume_base: Baseline mean log1p(volume).

    Returns:
        Array of shape (N, seq_len, 5) with Open, High, Low, Close, Volume.
    """
    idx = {name: i for i, name in enumerate(feature_cols)}
    missing = set(RETURN_FEATURES) - set(idx)
    if missing:
        raise ValueError(f"windows_to_ohlcv needs the 'returns' features; missing {missing}.")
    f = {name: windows[..., idx[name]] for name in RETURN_FEATURES}

    close = start_close * np.exp(np.cumsum(f["LogReturn"], axis=1))
    prev_close = np.concatenate([np.full_like(close[:, :1], start_close), close[:, :-1]], axis=1)
    open_ = prev_close * np.exp(f["OpenGap"])
    high = np.maximum(open_, close) * np.exp(np.clip(f["HighRange"], 0.0, None))
    low = np.minimum(open_, close) * np.exp(-np.clip(f["LowRange"], 0.0, None))
    volume = np.maximum(np.expm1(log_volume_base + f["LogVolumeRel"]), 1.0)
    return np.stack([open_, high, low, close, volume], axis=-1)
