"""Tests for the post-hoc LSTM suite and the PCA/t-SNE projections."""

import numpy as np
import pytest

from synfin.evaluation.metrics import compute_all_metrics
from synfin.evaluation.posthoc_rnn import (
    _real_split,
    rnn_discriminative_score,
    rnn_predictive_score,
)
from synfin.visualization.projections import compute_projections, plot_projections


def ar_windows(n, seq=16, phi=0.8, seed=0, sliding=False):
    """AR(1) windows: predictable next step. Sliding = stride-1 windows of one series."""
    rng = np.random.default_rng(seed)
    if sliding:
        x = np.zeros(n + seq)
        for t in range(1, len(x)):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        w = np.stack([x[i : i + seq] for i in range(n)])
    else:
        w = np.zeros((n, seq))
        w[:, 0] = rng.standard_normal(n) / np.sqrt(1 - phi**2)
        for t in range(1, seq):
            w[:, t] = phi * w[:, t - 1] + rng.standard_normal(n)
    return np.stack([w, np.abs(w)], axis=-1)


def test_real_split_purges_overlap_and_spans_the_period():
    rng = np.random.default_rng(0)
    tr, te = _real_split(1000, seq_len=16, sliding=True, test_frac=0.2, rng=rng)
    # No training window shares a day with a test window.
    assert np.abs(tr[:, None] - te[None, :]).min() >= 16
    # Test windows come from several blocks across the period, not one tail block.
    assert len(np.unique(te // 32)) >= 4 and te.min() < 500 < te.max()
    assert 0.15 < len(te) / 1000 < 0.25
    tr, te = _real_split(100, seq_len=16, sliding=False, test_frac=0.2, rng=rng)
    assert len(te) == 20 and not set(tr) & set(te)


def test_discriminator_near_chance_on_same_process():
    real = ar_windows(600, sliding=True, seed=0)
    synth = ar_windows(600, seed=1)
    out = rnn_discriminative_score(real, synth, epochs=8, seed=0)
    assert out["accuracy"] < 0.65
    assert out["score"] == pytest.approx(abs(out["accuracy"] - 0.5))


def test_discriminator_separates_different_processes():
    real = ar_windows(600, sliding=True, seed=0)
    noise = np.random.default_rng(2).standard_normal(real.shape) * real.std(axis=(0, 1))
    out = rnn_discriminative_score(real, noise, epochs=8, seed=0)
    assert out["accuracy"] > 0.8 and out["auc"] > 0.85


def test_predictive_ratio_near_one_for_same_process_and_high_for_noise():
    real_tr, real_ho = ar_windows(500, seed=0), ar_windows(200, seed=3)
    same = rnn_predictive_score(real_tr, ar_windows(500, seed=1), real_ho, epochs=8)
    noise = np.random.default_rng(4).standard_normal(real_tr.shape) * real_tr.std(axis=(0, 1))
    bad = rnn_predictive_score(real_tr, noise, real_ho, epochs=8)
    assert same["ratio"] < 1.15
    assert bad["ratio"] > same["ratio"] + 0.1
    assert same["mae_trtr"] < same["mae_naive"]


def test_projections_shapes_and_overlap(tmp_path):
    real, same = ar_windows(300, seed=0), ar_windows(300, seed=1)
    p = compute_projections(real, same, n_samples=200, seed=0)
    for k in ("real_pca", "synthetic_pca", "real_tsne", "synthetic_tsne"):
        assert p[k].shape == (200, 2)
    # Same distribution: PCA centroids nearly coincide relative to the spread.
    gap = np.linalg.norm(p["real_pca"].mean(0) - p["synthetic_pca"].mean(0))
    assert gap < 0.3 * p["real_pca"].std(0).mean()
    plot_projections(real, same, n_samples=100, savepath=tmp_path / "proj.png", reduce="flatten")
    assert (tmp_path / "proj.png").stat().st_size > 0
    with pytest.raises(ValueError):
        compute_projections(real, same, reduce="nope")


def test_compute_all_metrics_rnn_flag():
    real, synth, hold = (
        ar_windows(300, sliding=True),
        ar_windows(200, seed=1),
        ar_windows(80, seed=2),
    )
    report = compute_all_metrics(real, synth, real_holdout=hold, run_tstr=False, rnn_metrics=True)
    assert set(report["posthoc_rnn"]) == {"discriminative", "predictive"}
    assert "posthoc_rnn" not in report["realism_components"]
