"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.data.features import add_realized_volatility, add_features
from synfin.data.preprocess import compute_log_returns


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + abs(np.random.randn(n)),
            "Low": prices - abs(np.random.randn(n)),
            "Close": prices,
            "Volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )
    return compute_log_returns(df)


def test_realized_volatility(sample_ohlcv):
    """add_realized_volatility adds RealizedVol column."""
    result = add_realized_volatility(sample_ohlcv, window=10)
    assert "RealizedVol" in result.columns


def test_add_features_no_crash(sample_ohlcv):
    """add_features runs without raising even if ta is not installed."""
    result = add_features(
        sample_ohlcv,
        use_rsi=True,
        use_macd=True,
        use_bollinger=True,
        use_atr=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
