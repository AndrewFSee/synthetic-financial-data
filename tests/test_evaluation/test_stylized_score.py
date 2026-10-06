"""Tests for the noise-aware stylized-facts score."""

import numpy as np
import pytest

from synfin.evaluation.stylized_score import STATS, stylized_agreement, stylized_stats


def garch_t(n, seed, a=0.15, b=0.8, df=5):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    h, e = 1.0, 0.0
    for t in range(n):
        h = (1 - a - b) + a * e * e + b * h
        e = np.sqrt(h) * rng.standard_t(df) * np.sqrt((df - 2) / df)
        r[t] = 0.01 * e
    return r


def windows(series, seq=20):
    return np.stack([series[i : i + seq] for i in range(len(series) - seq + 1)])


@pytest.fixture(scope="module")
def real():
    return windows(garch_t(2000, 0))


@pytest.fixture(scope="module")
def same_process():
    return np.stack([garch_t(220, 1000 + i)[200:] for i in range(600)])


def test_stats_keys_and_volume_optional(real):
    s = stylized_stats(real)
    assert set(s) == set(STATS) - {"volume_volatility"}
    assert "volume_volatility" in stylized_stats(real, np.abs(real))


def test_same_process_scores_high(real, same_process):
    out = stylized_agreement(real, same_process, n_boot=40)
    assert out["score"] > 0.6
    assert all(abs(t["z"]) < 3 for t in out["terms"].values())


def test_time_shuffled_loses_clustering_but_keeps_tails(real):
    rng = np.random.default_rng(1)
    flat = real.ravel()
    shuffled = flat[rng.permutation(len(flat))][: 600 * 20].reshape(600, 20)
    terms = stylized_agreement(real, shuffled, n_boot=40)["terms"]
    assert terms["volatility_clustering"]["term"] < 0.3
    assert terms["regime_dispersion"]["term"] < 0.3
    assert terms["tail_weight"]["term"] > 0.5  # same marginal distribution


def test_iid_noise_scores_low(real):
    rng = np.random.default_rng(2)
    flat = real.ravel()
    noise = flat.mean() + flat.std() * rng.standard_normal((600, 20))
    assert stylized_agreement(real, noise, n_boot=40)["score"] < 0.2


def test_z_sign_and_volume_term(real, same_process):
    out = stylized_agreement(real, same_process, np.abs(real), -np.abs(same_process), n_boot=30)
    vv = out["terms"]["volume_volatility"]
    assert vv["z"] < -4 and vv["term"] == 0.0  # synthetic volume anti-correlated with |r|
