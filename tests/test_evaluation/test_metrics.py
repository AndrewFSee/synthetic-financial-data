"""Tests for evaluation metrics."""

import json

import numpy as np
import pytest

from synfin.evaluation.discriminative import discriminative_score, window_features
from synfin.evaluation.metrics import compute_all_metrics
from synfin.evaluation.privacy import (
    collapse_diagnostics,
    distance_to_closest_record,
    membership_inference_risk,
    nearest_neighbor_distance_ratio,
)
from synfin.evaluation.statistical_tests import acf_comparison, ks_test, mmd_rbf, pooled_acf
from synfin.evaluation.stylized_facts import (
    check_all_stylized_facts,
    check_fat_tails,
    check_leverage_effect,
    check_volatility_clustering,
    extreme_move_check,
)
from synfin.evaluation.tstr import tstr_benchmark


def garch_returns(n: int, seed: int = 0, alpha: float = 0.1, beta: float = 0.85) -> np.ndarray:
    """Simple GARCH(1,1) returns with clear volatility clustering."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    h = 1.0
    eps = 0.0
    for t in range(n):
        h = (1 - alpha - beta) + alpha * eps**2 + beta * h
        eps = np.sqrt(h) * rng.standard_normal()
        r[t] = 0.01 * eps
    return r


def windows_of(series: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Stride-1 sliding windows, like the real preprocessing produces."""
    return np.stack([series[i : i + seq_len] for i in range(len(series) - seq_len + 1)])


@pytest.fixture
def random_data():
    np.random.seed(42)
    real = np.random.randn(200, 5)
    synthetic = np.random.randn(150, 5)
    return real, synthetic


# --- statistical tests ------------------------------------------------------


def test_ks_test_same_distribution():
    """KS on identical distributions: small statistic, valid p-values."""
    rng = np.random.default_rng(0)
    results = ks_test(rng.standard_normal((500, 3)), rng.standard_normal((500, 3)))
    for res in results.values():
        assert res["statistic"] < 0.15
        assert 0.0 <= res["p_value"] <= 1.0


def test_ks_test_different_distributions():
    """KS on shifted distributions rejects."""
    rng = np.random.default_rng(1)
    results = ks_test(rng.standard_normal((500, 2)), rng.standard_normal((500, 2)) + 10.0)
    for res in results.values():
        assert res["p_value"] < 0.05
        assert res["reject"]


def test_mmd_same_distribution():
    """Unbiased MMD² between samples of one distribution is ~0."""
    rng = np.random.default_rng(2)
    mmd = mmd_rbf(rng.standard_normal((200, 5)), rng.standard_normal((200, 5)))
    assert abs(mmd) < 0.05


def test_mmd_different_distributions():
    """MMD² between shifted distributions is clearly positive."""
    rng = np.random.default_rng(3)
    mmd = mmd_rbf(rng.standard_normal((100, 5)), rng.standard_normal((100, 5)) + 5.0)
    assert mmd > 0.05


def test_mmd_handles_large_flattened_windows():
    """3-D window inputs with many dims run without an N*M*D intermediate."""
    rng = np.random.default_rng(4)
    mmd = mmd_rbf(rng.standard_normal((3000, 30, 8)), rng.standard_normal((1000, 30, 8)))
    assert np.isfinite(mmd)


def test_acf_comparison(random_data):
    """acf_comparison returns correct structure for a single long series."""
    real, synthetic = random_data
    results = acf_comparison(real, synthetic, max_lag=10)
    assert len(results) == 5
    for vals in results.values():
        assert len(vals["real_acf"]) == 11
        assert "mae" in vals


def test_pooled_acf_recovers_ar_coefficient():
    """Within-window pooling recovers the AR(1) coefficient from sliding windows."""
    rng = np.random.default_rng(5)
    x = np.zeros(5000)
    for t in range(1, len(x)):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
    acf = pooled_acf(windows_of(x, 30), max_lag=3)
    assert acf[1] == pytest.approx(0.5, abs=0.05)
    assert acf[2] == pytest.approx(0.25, abs=0.05)


def test_pooled_acf_caps_lag_at_window_length():
    acf = pooled_acf(np.random.randn(10, 8), max_lag=20)
    assert len(acf) == 7  # lags 0..6


# --- stylized facts ----------------------------------------------------------


def test_check_fat_tails():
    """Laplace returns are fat-tailed; Gaussian returns are not."""
    rng = np.random.default_rng(42)
    assert check_fat_tails(rng.laplace(0, 1, 5000))["is_fat_tailed"]
    assert not check_fat_tails(rng.standard_normal(5000))["is_fat_tailed"]


def test_fat_tail_robust_measures():
    """A single huge outlier dominates raw kurtosis but barely moves the robust measures."""
    rng = np.random.default_rng(5)
    r = rng.standard_normal(20000)
    spiked = r.copy()
    spiked[0] = 40.0
    base, spike = check_fat_tails(r), check_fat_tails(spiked)
    assert spike["excess_kurtosis"] > 50 * max(abs(base["excess_kurtosis"]), 0.1)
    assert spike["max_sd"] > 30
    assert abs(spike["trimmed_excess_kurtosis"] - base["trimmed_excess_kurtosis"]) < 0.1
    assert abs(spike["q999_sd"] - base["q999_sd"]) < 0.2
    assert base["q99_sd"] == pytest.approx(2.576, abs=0.1)  # Gaussian 99% two-sided


def test_extreme_move_check_is_size_matched():
    """Same distribution: calibrated even though synthetic has 15x more days."""
    rng = np.random.default_rng(6)
    real = windows_of(rng.standard_normal(2000), 30)  # stride-1, 2000 unique days
    synth = rng.standard_normal((1000, 30))  # 30,000 independent days
    out = extreme_move_check(real, synth)
    assert out["n_real_days"] == 2000
    assert abs(out["n_matched_days"] - 2000) <= 30
    assert np.abs(synth).max() / out["real_max"] > 1.1  # naive raw-max comparison misleads
    assert 0.85 < out["max_ratio_median"] < 1.2
    assert 0.2 < out["p_exceed_real_max"] < 0.8


def test_extreme_move_check_flags_thin_tails():
    rng = np.random.default_rng(7)
    real = windows_of(rng.standard_t(3, 3000) / np.sqrt(3), 30)
    synth = rng.standard_normal((1000, 30))  # same variance, Gaussian tails
    out = extreme_move_check(real, synth)
    assert out["max_ratio_median"] < 0.8
    assert out["p_exceed_real_max"] < 0.2


def test_volatility_clustering_detected_in_windows():
    """GARCH windows show clustering; i.i.d. windows do not."""
    garch = windows_of(garch_returns(4000), 30)
    iid = windows_of(np.random.default_rng(1).standard_normal(4000) * 0.01, 30)
    assert check_volatility_clustering(garch)["has_clustering"]
    assert not check_volatility_clustering(iid)["has_clustering"]


def test_stylized_facts_invariant_to_window_offset_scaling():
    """The old bug: |minmax-scaled return| is not volatility. Unscaled returns are needed."""
    r = windows_of(garch_returns(4000), 30)
    scaled = (r - r.min()) / (r.max() - r.min())  # every value >= 0
    assert check_volatility_clustering(r)["mean_abs_return_acf"] > 0.05
    # On min-max scaled data |x| == x, so the "clustering" signal collapses.
    assert (
        check_volatility_clustering(scaled)["mean_abs_return_acf"]
        < check_volatility_clustering(r)["mean_abs_return_acf"]
    )


def test_leverage_effect_sign():
    """Negative returns followed by higher volatility give a negative leverage correlation."""
    rng = np.random.default_rng(6)
    r = np.empty(5000)
    r[0] = 0.0
    for t in range(1, len(r)):
        vol = 1.0 + 0.8 * max(-r[t - 1], 0.0)
        r[t] = vol * rng.standard_normal()
    out = check_leverage_effect(windows_of(r, 20))
    assert out["mean_leverage_corr"] < -0.02
    assert out["has_leverage_effect"]


def test_check_all_stylized_facts_keys():
    r = windows_of(garch_returns(1000), 20)
    out = check_all_stylized_facts(r, volume=np.abs(r))
    assert set(out) == {
        "fat_tails",
        "volatility_clustering",
        "leverage_effect",
        "volume_volatility_correlation",
    }


# --- TSTR --------------------------------------------------------------------


def _features(series: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Windows with a LogReturn column (index 0) and |r| as a second feature."""
    w = windows_of(series, seq_len)
    return np.stack([w, np.abs(w)], axis=-1)


def test_tstr_works_on_minmax_scaled_returns():
    """The old bug: with min-max scaling every return is > 0, leaving one class."""
    w = _features(garch_returns(2000))
    lo, hi = w.min(axis=(0, 1)), w.max(axis=(0, 1))
    scaled = (w - lo) / (hi - lo)
    assert (scaled[:, :, 0] >= 0).all()
    for task in ("volatility", "direction"):
        out = tstr_benchmark(scaled, scaled, return_idx=0, task=task)
        assert "skipped" not in out["trtr"]
        assert "tstr_gap" in out


def test_tstr_volatility_task_is_learnable():
    """Volatility clustering makes next-step |r| predictable on real-like data."""
    real = _features(garch_returns(4000, seed=0, alpha=0.15, beta=0.8))
    synth = _features(garch_returns(4000, seed=1, alpha=0.15, beta=0.8))
    out = tstr_benchmark(real, synth, return_idx=0, task="volatility")
    assert out["trtr"]["auc"] > 0.55
    assert out["tstr"]["auc"] > 0.55


def test_tstr_uses_given_test_set():
    real = _features(garch_returns(1500, seed=0))
    test = _features(garch_returns(600, seed=2))
    out = tstr_benchmark(real, real, return_idx=0, real_test=test)
    assert "auc" in out["trtr"]


# --- privacy -----------------------------------------------------------------


def test_nndr_shape(random_data):
    real, synthetic = random_data
    result = nearest_neighbor_distance_ratio(real, synthetic)
    assert result["nndr"] >= 0
    assert "median_dist_holdout_to_real" in result


def test_dcr_shape(random_data):
    real, synthetic = random_data
    result = distance_to_closest_record(real, synthetic)
    assert result["min_dcr"] >= 0


def test_memorization_detected_for_copies():
    """Synthetic samples copied from the training data are flagged."""
    rng = np.random.default_rng(7)
    train = rng.standard_normal((500, 10))
    holdout = rng.standard_normal((200, 10))
    copies = train[:200] + rng.standard_normal((200, 10)) * 1e-3
    mi = membership_inference_risk(train, copies, holdout=holdout)
    nndr = nearest_neighbor_distance_ratio(train, copies, holdout=holdout)
    assert mi["memorization_rate"] > 0.5
    assert nndr["nndr"] < 0.1


def test_no_memorization_for_fresh_samples():
    """Fresh samples from the same distribution behave like the holdout."""
    rng = np.random.default_rng(8)
    train = rng.standard_normal((500, 10))
    holdout = rng.standard_normal((400, 10))
    fresh = rng.standard_normal((400, 10))
    mi = membership_inference_risk(train, fresh, holdout=holdout)
    nndr = nearest_neighbor_distance_ratio(train, fresh, holdout=holdout)
    assert mi["memorization_rate"] < 0.15
    assert nndr["nndr"] == pytest.approx(1.0, abs=0.1)
    assert not collapse_diagnostics(train, fresh, holdout=holdout)["collapsed"]


def test_collapse_is_not_mistaken_for_memorization():
    """Samples squeezed toward the data centre are near many records, copying none.

    Raw distance-to-closest-record calls this memorization; the d1/d2 test does
    not, and the collapse diagnostics flag it instead.
    """
    rng = np.random.default_rng(10)
    train = rng.standard_normal((800, 10))
    holdout = rng.standard_normal((300, 10))
    collapsed = 0.1 * rng.standard_normal((300, 10))
    mi = membership_inference_risk(train, collapsed, holdout=holdout)
    col = collapse_diagnostics(train, collapsed, holdout=holdout)
    assert mi["median_dcr"] < mi["holdout_median_dcr"]  # raw DCR would cry memorization
    assert mi["memorization_rate"] < 0.1
    assert col["collapsed"]
    assert col["dispersion_ratio"] < 0.5


def test_privacy_verdicts_in_report():
    rng = np.random.default_rng(11)
    real = rng.standard_normal((400, 6, 2))
    holdout = rng.standard_normal((150, 6, 2))
    cases = {
        "ok": rng.standard_normal((300, 6, 2)),
        "collapse": 0.1 * rng.standard_normal((300, 6, 2)),
        "memorization": real[:300] + 1e-3 * rng.standard_normal((300, 6, 2)),
    }
    for verdict, synth in cases.items():
        report = compute_all_metrics(real, synth, real_holdout=holdout, run_tstr=False)
        assert report["privacy"]["verdict"] == verdict
    assert report["realism_components"]["privacy"] < 0.1  # memorization case


# --- discriminative score ----------------------------------------------------


def test_window_features_shape():
    w = _features(garch_returns(300))
    assert window_features(w).shape == (len(w), 8)
    assert window_features(w, return_idx=0).shape == (len(w), 11)


def test_discriminative_same_process_is_indistinguishable():
    real = _features(garch_returns(3000, seed=0))
    synth = _features(garch_returns(3000, seed=1))
    assert discriminative_score(real, synth, return_idx=0)["auc"] < 0.6


def test_discriminative_catches_iid_noise_with_matching_moments():
    """KS/MMD-style checks miss this; the dynamics classifier does not."""
    real = _features(garch_returns(3000, seed=0, alpha=0.15, beta=0.8))
    flat = real.reshape(-1, real.shape[-1])
    rng = np.random.default_rng(1)
    noise = flat.mean(0) + flat.std(0) * rng.standard_normal((1000,) + real.shape[1:])
    out = discriminative_score(real, noise, return_idx=0)
    assert out["auc"] > 0.8
    assert out["score"] < 0.4


# --- compute_all_metrics -----------------------------------------------------


def test_compute_all_metrics_end_to_end(tmp_path):
    real = _features(garch_returns(1500, seed=0))
    synth = _features(garch_returns(800, seed=1))
    holdout = _features(garch_returns(400, seed=2))
    report = compute_all_metrics(
        real,
        synth,
        feature_names=["LogReturn", "AbsReturn"],
        real_holdout=holdout,
        output_dir=str(tmp_path),
    )
    assert "stylized_facts_real" in report and "tstr" in report
    assert set(report["realism_components"]) == {
        "ks",
        "mmd",
        "tstr",
        "privacy",
        "discriminative",
        "stylized",
    }
    assert all(0.0 <= v <= 1.0 for v in report["realism_components"].values())
    assert 0.0 <= report["realism_score"] <= 1.0
    json.loads((tmp_path / "evaluation_report.json").read_text())


def test_compute_all_metrics_without_names_skips_return_metrics():
    rng = np.random.default_rng(9)
    report = compute_all_metrics(
        rng.standard_normal((120, 10, 3)), rng.standard_normal((80, 10, 3)), run_tstr=True
    )
    assert "stylized_facts_real" not in report
    assert "tstr" not in report
    assert "mmd" in report


def test_compute_all_metrics_rejects_wrong_names():
    with pytest.raises(ValueError):
        compute_all_metrics(np.zeros((60, 5, 2)), np.zeros((60, 5, 2)), feature_names=["a"])
