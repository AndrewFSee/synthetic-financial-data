"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from synfin.data.preprocess import (
    RETURN_FEATURES,
    apply_scaling,
    compute_dollar_volume,
    compute_log_returns,
    compute_log_volume,
    compute_return_features,
    create_windows,
    invert_scaling,
    normalize,
    preprocess,
    train_val_test_split,
    windows_to_ohlcv,
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
    for key in ["train", "val", "test", "scaler", "scaler_params", "feature_cols"]:
        assert key in result
    assert result["feature_cols"] == RETURN_FEATURES
    assert result["train"].ndim == 3
    assert result["train"].shape[1:] == (20, len(RETURN_FEATURES))


def test_preprocess_has_no_lookahead(sample_df):
    """Changing the test-period rows must not change the training windows."""
    base = preprocess(sample_df, window_size=20)
    shocked = sample_df.copy()
    shocked.iloc[-40:, :4] *= 5.0  # huge price jump late in the sample
    shocked.iloc[-40:, 4] *= 50.0
    other = preprocess(shocked, window_size=20)
    np.testing.assert_array_equal(base["train"], other["train"])
    assert base["scaler_params"] == other["scaler_params"]


def test_preprocess_scaler_fit_on_train_rows_only(sample_df):
    """Z-scored training windows have ~zero mean / unit std per feature."""
    result = preprocess(sample_df, window_size=20, normalization="zscore")
    first_rows = np.concatenate([result["train"][0], result["train"][1:, -1]])  # each row once
    np.testing.assert_allclose(first_rows.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(first_rows.std(axis=0), 1.0, atol=1e-3)


def test_preprocess_windows_do_not_cross_splits(sample_df):
    """Each split is windowed separately: N_rows - window + 1 windows per split."""
    result = preprocess(sample_df, window_size=20, train_ratio=0.7, val_ratio=0.15)
    n_rows = len(compute_return_features(sample_df).dropna(subset=RETURN_FEATURES))
    n_train = int(n_rows * 0.7)
    n_val = int(n_rows * 0.15)
    assert len(result["train"]) == n_train - 19
    assert len(result["val"]) == n_val - 19
    assert len(result["test"]) == n_rows - n_train - n_val - 19


def test_preprocess_levels_feature_set(sample_df):
    result = preprocess(sample_df, window_size=20, feature_set="levels", normalization="minmax")
    assert "Close" in result["feature_cols"]
    assert result["train"].min() >= -1e-6 and result["train"].max() <= 1 + 1e-6


def test_preprocess_unknown_feature_set(sample_df):
    with pytest.raises(ValueError, match="feature_set"):
        preprocess(sample_df, feature_set="nope")


def test_scaling_roundtrip(sample_df):
    """apply_scaling / invert_scaling reproduce the sklearn scaler both ways."""
    for method in ("zscore", "minmax"):
        result = preprocess(sample_df, window_size=20, normalization=method)
        w = result["train"]
        raw = invert_scaling(w, result["scaler_params"])
        np.testing.assert_allclose(apply_scaling(raw, result["scaler_params"]), w, atol=1e-5)
        sk = result["scaler"].inverse_transform(w.reshape(-1, w.shape[-1])).reshape(w.shape)
        np.testing.assert_allclose(raw, sk, rtol=1e-5, atol=1e-6)


def test_return_features_rebuild_ohlcv(sample_df):
    """windows_to_ohlcv inverts the returns feature set (prices exactly)."""
    df = sample_df.copy()
    df["High"] = np.maximum(df["High"], df[["Open", "Close"]].max(axis=1))
    df["Low"] = np.minimum(df["Low"], df[["Open", "Close"]].min(axis=1))
    feats = compute_return_features(df).dropna(subset=RETURN_FEATURES)
    window = feats[RETURN_FEATURES].to_numpy()[None, :30]
    start = df["Close"].shift(1).loc[feats.index[0]]
    ohlcv = windows_to_ohlcv(window, RETURN_FEATURES, start_close=start)[0]
    expected = df.loc[feats.index[:30], ["Open", "High", "Low", "Close"]].to_numpy()
    np.testing.assert_allclose(ohlcv[:, :4], expected, rtol=1e-8)


def test_windows_to_ohlcv_always_valid_bars():
    """Even noisy generated features (negative ranges) give valid OHLC bars."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((50, 20, 5)) * 0.02
    ohlcv = windows_to_ohlcv(w, RETURN_FEATURES)
    o, h, lo, c, v = np.moveaxis(ohlcv, -1, 0)
    assert (lo <= np.minimum(o, c) + 1e-12).all()
    assert (h >= np.maximum(o, c) - 1e-12).all()
    assert (v >= 1).all() and (ohlcv[..., :4] > 0).all()
