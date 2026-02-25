"""Tests for data download module."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.data.download import download_ohlcv, load_ohlcv


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame."""
    import numpy as np
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame(
        {
            "Open": prices + np.random.randn(100) * 0.1,
            "High": prices + abs(np.random.randn(100)) * 0.5,
            "Low": prices - abs(np.random.randn(100)) * 0.5,
            "Close": prices,
            "Volume": np.random.randint(1000, 10000, 100).astype(float),
        },
        index=dates,
    )


def test_load_ohlcv_missing_file():
    """load_ohlcv returns None when file does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_ohlcv("FAKE", data_dir=tmpdir)
        assert result is None


def test_download_ohlcv_saves_parquet(sample_ohlcv_df):
    """download_ohlcv saves a parquet file when given valid mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("synfin.data.download.yf.download", return_value=sample_ohlcv_df):
            result = download_ohlcv(
                tickers=["AAPL"],
                start="2020-01-01",
                end="2020-12-31",
                output_dir=tmpdir,
                save_format="parquet",
            )

        assert "AAPL" in result
        assert len(result["AAPL"]) > 0
        assert (Path(tmpdir) / "AAPL_1d.parquet").exists()


def test_download_ohlcv_csv(sample_ohlcv_df):
    """download_ohlcv saves a CSV file when format='csv'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("synfin.data.download.yf.download", return_value=sample_ohlcv_df):
            result = download_ohlcv(
                tickers=["MSFT"],
                output_dir=tmpdir,
                save_format="csv",
            )
        assert "MSFT" in result
        assert (Path(tmpdir) / "MSFT_1d.csv").exists()
