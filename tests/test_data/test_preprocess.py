"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.data.preprocess import (
    compute_log_returns,
    compute_log_volume,
    compute_dollar_volume,
    normalize,
    create_windows,
    train_val_test_split,
    preprocess,
)


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame."""
    np.random.seed(0)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": prices + np.random.randn(n) * 0.1,
            "High": prices + abs(np.random.randn(n)) * 0.5,
            "Low": prices - abs(np.random.randn(n)) * 0.5,
            "Close": prices,
            "Volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )


def test_log_returns(sample_df):
    """compute_log_returns adds LogReturn column."""
    result = compute_log_returns(sample_df)
    assert "LogReturn" in result.columns
    assert result["LogReturn"].isna().sum() == 0


def test_log_volume(sample_df):
    """compute_log_volume adds LogVolume column."""
    result = compute_log_volume(sample_df)
    assert "LogVolume" in result.columns
    assert (result["LogVolume"] >= 0).all()


def test_dollar_volume(sample_df):
    """compute_dollar_volume adds DollarVolume column."""
    result = compute_dollar_volume(sample_df)
    assert "DollarVolume" in result.columns
    assert (result["DollarVolume"] > 0).all()


def test_normalize_minmax(sample_df):
    """Normalized values should be in [0, 1] for minmax."""
    df_norm, scaler = normalize(sample_df, method="minmax", feature_cols=["Close"])
    assert df_norm["Close"].min() >= -1e-6
    assert df_norm["Close"].max() <= 1 + 1e-6


def test_normalize_zscore(sample_df):
    """Z-score normalization should produce near-zero mean."""
    df_norm, scaler = normalize(sample_df, method="zscore", feature_cols=["Close"])
    assert abs(df_norm["Close"].mean()) < 0.1


def test_normalize_invalid_method(sample_df):
    """normalize raises ValueError for unknown method."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize(sample_df, method="invalid")


def test_create_windows(sample_df):
    """create_windows produces correct shape."""
    df_with_returns = compute_log_returns(sample_df)
    feature_cols = ["Close", "Volume"]
    windows = create_windows(df_with_returns, window_size=30, feature_cols=feature_cols)
    assert windows.ndim == 3
    assert windows.shape[1] == 30
    assert windows.shape[2] == len(feature_cols)


def test_create_windows_too_short():
    """create_windows raises ValueError when DataFrame is too short."""
    short_df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="too short"):
        create_windows(short_df, window_size=30, feature_cols=["Close"])


def test_train_val_test_split():
    """Split produces correct proportions."""
    windows = np.random.randn(100, 30, 5).astype(np.float32)
    train, val, test = train_val_test_split(windows, train_ratio=0.7, val_ratio=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_preprocess_pipeline(sample_df):
    """Full preprocess pipeline produces correct structure."""
    result = preprocess(sample_df, window_size=20, train_ratio=0.7, val_ratio=0.15)
    assert "train" in result
    assert "val" in result
    assert "test" in result
    assert "scaler" in result
    assert "feature_cols" in result
    assert result["train"].ndim == 3
    assert result["train"].shape[1] == 20
