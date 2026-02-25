"""Tests for OHLCV constraint enforcement."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.constraints.ohlcv import (
    enforce_ohlc_constraints,
    enforce_volume_constraints,
    enforce_price_continuity,
    enforce_positive_prices,
    apply_all_constraints,
)


@pytest.fixture
def invalid_ohlcv():
    """DataFrame with intentionally invalid OHLCV values."""
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 98.0, 105.0],
            "High": [99.0, 101.0, 97.0, 104.0],   # High < Open (invalid)
            "Low": [101.0, 103.0, 99.0, 106.0],   # Low > Open (invalid)
            "Close": [100.5, 102.5, 98.5, 105.5],
            "Volume": [-100.0, 0.0, 500.0, 1000.0],  # Negative/zero (invalid)
        }
    )


def test_enforce_ohlc_high_gte_max_oc(invalid_ohlcv):
    """After constraint, High >= max(Open, Close) for all rows."""
    fixed = enforce_ohlc_constraints(invalid_ohlcv)
    oc_max = fixed[["Open", "Close"]].max(axis=1)
    assert (fixed["High"] >= oc_max - 1e-10).all()


def test_enforce_ohlc_low_lte_min_oc(invalid_ohlcv):
    """After constraint, Low <= min(Open, Close) for all rows."""
    fixed = enforce_ohlc_constraints(invalid_ohlcv)
    oc_min = fixed[["Open", "Close"]].min(axis=1)
    assert (fixed["Low"] <= oc_min + 1e-10).all()


def test_enforce_volume_positive(invalid_ohlcv):
    """After constraint, all volume values are > 0."""
    fixed = enforce_volume_constraints(invalid_ohlcv)
    assert (fixed["Volume"] > 0).all()


def test_enforce_volume_round_to_int():
    """Volume is rounded to integer when round_to_int=True."""
    df = pd.DataFrame({"Volume": [1.7, 2.3, 100.9]})
    fixed = enforce_volume_constraints(df, round_to_int=True)
    assert fixed["Volume"].dtype in [int, np.int64, np.int32]


def test_enforce_price_continuity():
    """enforce_price_continuity clips unrealistic jumps."""
    df = pd.DataFrame({"Close": [100.0, 200.0, 201.0]})  # 100% jump
    fixed = enforce_price_continuity(df, max_gap_pct=0.20)
    pct_change = abs(fixed["Close"].iloc[1] - fixed["Close"].iloc[0]) / fixed["Close"].iloc[0]
    assert pct_change <= 0.20 + 1e-6


def test_enforce_positive_prices():
    """enforce_positive_prices replaces non-positive values."""
    df = pd.DataFrame({
        "Open": [-1.0, 0.0, 100.0],
        "High": [10.0, 0.0, 110.0],
        "Low": [-5.0, 0.0, 90.0],
        "Close": [0.0, 0.0, 100.0],
    })
    fixed = enforce_positive_prices(df)
    for col in ["Open", "High", "Low", "Close"]:
        assert (fixed[col] > 0).all()


def test_apply_all_constraints(invalid_ohlcv):
    """apply_all_constraints applies all fixes without error."""
    fixed = apply_all_constraints(invalid_ohlcv)
    assert isinstance(fixed, pd.DataFrame)
    assert len(fixed) == len(invalid_ohlcv)
    assert (fixed["Volume"] > 0).all()
    oc_max = fixed[["Open", "Close"]].max(axis=1)
    assert (fixed["High"] >= oc_max - 1e-10).all()
