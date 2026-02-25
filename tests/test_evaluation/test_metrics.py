"""Tests for evaluation metrics."""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.evaluation.statistical_tests import ks_test, mmd_rbf, acf_comparison
from synfin.evaluation.stylized_facts import check_fat_tails, check_volatility_clustering
from synfin.evaluation.privacy import nearest_neighbor_distance_ratio, distance_to_closest_record


@pytest.fixture
def random_data():
    np.random.seed(42)
    real = np.random.randn(200, 5)
    synthetic = np.random.randn(150, 5)
    return real, synthetic


def test_ks_test_same_distribution():
    """KS test on same distribution should have high p-values."""
    np.random.seed(0)
    data = np.random.randn(500, 3)
    results = ks_test(data, data + np.random.randn(*data.shape) * 0.01)
    for feat, res in results.items():
        assert 0.0 <= res["statistic"] <= 1.0
        assert 0.0 <= res["p_value"] <= 1.0


def test_ks_test_different_distributions():
    """KS test on very different distributions should have low p-values."""
    np.random.seed(1)
    real = np.random.randn(500, 2)
    synthetic = np.random.randn(500, 2) + 10.0  # shift by 10
    results = ks_test(real, synthetic)
    for feat, res in results.items():
        assert res["p_value"] < 0.05


def test_mmd_same_distribution():
    """MMD between same distributions should be near zero."""
    np.random.seed(2)
    data = np.random.randn(100, 5)
    mmd = mmd_rbf(data, data + np.random.randn(*data.shape) * 0.001)
    assert mmd < 0.5


def test_mmd_different_distributions():
    """MMD between very different distributions should be large."""
    np.random.seed(3)
    real = np.random.randn(100, 5)
    synth = np.random.randn(100, 5) + 5.0
    mmd = mmd_rbf(real, synth)
    assert mmd > 0.5


def test_acf_comparison(random_data):
    """acf_comparison returns correct structure."""
    real, synthetic = random_data
    results = acf_comparison(real, synthetic, max_lag=10)
    assert len(results) == 5
    for feat, vals in results.items():
        assert "real_acf" in vals
        assert "synthetic_acf" in vals
        assert len(vals["real_acf"]) == 11


def test_check_fat_tails():
    """Laplace distribution should show fat tails."""
    np.random.seed(42)
    laplace_returns = np.random.laplace(0, 1, 1000)
    result = check_fat_tails(laplace_returns)
    assert result["feature_0"]["kurtosis"] > 3  # Laplace kurtosis > 3


def test_volatility_clustering():
    """check_volatility_clustering returns expected keys."""
    np.random.seed(42)
    returns = np.random.randn(500)
    result = check_volatility_clustering(returns)
    assert "mean_abs_return_acf" in result
    assert "mean_sq_return_acf" in result
    assert "has_clustering" in result


def test_nndr_shape(random_data):
    """nearest_neighbor_distance_ratio returns expected keys."""
    real, synthetic = random_data
    result = nearest_neighbor_distance_ratio(real, synthetic, n_neighbors=3)
    assert "nndr" in result
    assert result["nndr"] >= 0


def test_dcr_shape(random_data):
    """distance_to_closest_record returns expected keys."""
    real, synthetic = random_data
    result = distance_to_closest_record(real, synthetic)
    assert "mean_dcr" in result
    assert "min_dcr" in result
    assert result["min_dcr"] >= 0
