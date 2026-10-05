"""Tests for the structural-break data generator, exporters and evaluation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from synfin.structural_breaks import (
    auc_report,
    baseline_scores,
    generate,
    get_preset,
    save_adia,
    to_adia_offline,
    to_adia_realtime,
    ts_auc,
)
from synfin.structural_breaks.presets import LayoutConfig


def _small(preset: str, **layout):
    """A preset with short series so tests run fast."""
    cfg = get_preset(preset)
    cfg.layout = replace(cfg.layout, hist_len=(200, 300), online_len=(50, 120), **layout)
    return cfg


@pytest.fixture(scope="module")
def offline_ds():
    return generate(300, _small("adia_offline", break_prob=0.5), seed=0)


@pytest.fixture(scope="module")
def realtime_ds():
    return generate(300, _small("adia_realtime"), seed=0)


def test_lengths_and_labels(offline_ds):
    """Series lengths, taus and labels are mutually consistent."""
    meta = offline_ds.meta
    assert len(offline_ds) == 300
    for s, (_, row) in zip(offline_ds.series, meta.iterrows()):
        assert len(s) == row.n_hist + row.n_online
        assert np.isfinite(s).all()
    assert ((meta.tau >= 0) == meta.has_break).all()
    assert (meta.loc[meta.has_break, "tau_index"] == 0).all()
    assert (meta.loc[~meta.has_break, "break_type"] == "none").all()
    assert (meta.loc[meta.has_break, "break_type"] != "none").all()


def test_deterministic_given_seed():
    """The same seed reproduces the dataset exactly; a new seed changes it."""
    cfg = _small("adia_offline")
    a, b = generate(20, cfg, seed=7), generate(20, cfg, seed=7)
    c = generate(20, cfg, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a.series, b.series))
    pd.testing.assert_frame_equal(a.meta, b.meta)
    assert not np.array_equal(a.series[0], c.series[0])


def test_realtime_tau_inside_online(realtime_ds):
    """Real-time breaks fall inside the online segment; values are z-scored."""
    meta = realtime_ds.meta
    br = meta[meta.has_break]
    assert (br.tau_index >= 0).all() and (br.tau_index < br.n_online).all()
    assert (br.tau == br.n_hist + br.tau_index).all()
    for s, n in zip(realtime_ds.series, meta.n_hist):
        assert abs(s[:n].mean()) < 1e-8
        assert abs(s[:n].std() - 1.0) < 1e-8


def test_null_series_parameters_unchanged(offline_ds):
    """No-break series have identical pre and post parameters."""
    meta = offline_ds.meta[~offline_ds.meta.has_break]
    for k in ("mu", "log_sigma", "phi", "alpha", "persistence", "nu_inv"):
        np.testing.assert_array_equal(meta[f"pre_{k}"], meta[f"post_{k}"])


@pytest.mark.parametrize("kind", ["mean", "variance", "ar", "tails", "garch"])
def test_each_break_type_changes_its_parameter(kind):
    """Forcing one break type moves exactly the parameter it is meant to move."""
    cfg = _small("adia_offline", break_prob=1.0)
    cfg.breaks = replace(cfg.breaks, type_weights={kind: 1.0})
    meta = generate(60, cfg, seed=1).meta
    param = {
        "mean": "mu",
        "variance": "log_sigma",
        "ar": "phi",
        "tails": "nu_inv",
        "garch": "alpha",
    }[kind]
    assert (meta.break_type == kind).all()
    assert (meta[f"post_{param}"] != meta[f"pre_{param}"]).mean() > 0.9
    assert (meta.effect > 0).mean() > 0.9
    others = {"mu", "log_sigma", "phi", "nu_inv"} - {param}
    for k in others:
        np.testing.assert_array_equal(meta[f"pre_{k}"], meta[f"post_{k}"])


def test_large_variance_break_is_visible():
    """With exaggerated magnitude, variance breaks are near-perfectly detectable."""
    cfg = _small("adia_offline", break_prob=0.5)
    cfg.breaks = replace(cfg.breaks, type_weights={"variance": 1.0}, magnitude_scale=6.0)
    cfg.process = replace(cfg.process, slow_vol_prob=0.0)
    cfg.decoys = replace(cfg.decoys, prob=0.0)
    ds = generate(200, cfg, seed=2)
    report = auc_report(baseline_scores(ds)["log_var_ratio"], ds.meta)
    assert report.loc[report.slice == "overall", "auc"].item() > 0.9


def test_bootstrap_base_process():
    """A bootstrap source restricts breaks to mean/variance and yields finite data."""
    rng = np.random.default_rng(0)
    source = rng.standard_t(4, size=2000) * 0.01
    cfg = _small("adia_offline", break_prob=1.0)
    ds = generate(50, cfg, seed=3, bootstrap_source=source)
    types = set("+".join(ds.meta.break_type).split("+"))
    assert types <= {"mean", "variance"}
    assert all(np.isfinite(s).all() for s in ds.series)


def test_offline_format(offline_ds, tmp_path):
    """Offline export matches the ADIA 2025 layout."""
    X, y = to_adia_offline(offline_ds)
    assert X.index.names == ["id", "time"]
    assert list(X.columns) == ["value", "period"]
    assert set(X["period"].unique()) == {0, 1}
    assert y.columns.tolist() == ["structural_breakpoint"]
    assert y["structural_breakpoint"].dtype == bool
    first = offline_ds.meta.iloc[0]
    periods = X.loc[0, "period"].to_numpy()
    assert (periods[: first.n_hist] == 0).all() and (periods[first.n_hist :] == 1).all()

    save_adia(offline_ds, tmp_path, layout="offline")
    assert pd.read_parquet(tmp_path / "X_train.parquet").shape == X.shape
    assert (tmp_path / "meta_train.parquet").exists()


def test_offline_format_rejects_realtime(realtime_ds):
    with pytest.raises(ValueError):
        to_adia_offline(realtime_ds)


def test_realtime_format(realtime_ds):
    """Real-time export: periods 1/2, tau index table, per-step targets."""
    X, y_index, y_steps = to_adia_realtime(realtime_ds)
    assert set(X["period"].unique()) == {1, 2}
    assert y_index.columns.tolist() == ["tau_index", "tau"]
    assert len(y_steps) == (X["period"] == 2).sum()
    meta = realtime_ds.meta
    sid = int(meta.index[meta.has_break][0])
    target = y_steps.loc[sid, "target"].to_numpy()
    assert target.sum() == meta.loc[sid, "n_online"] - meta.loc[sid, "tau_index"]
    assert target[meta.loc[sid, "tau_index"]] == 1


def test_ts_auc_oracle_and_constant(realtime_ds):
    """An oracle scores 1.0 and a time-only score scores 0.5."""
    meta = realtime_ds.meta
    oracle, constant = {}, {}
    for i, row in meta.iterrows():
        steps = np.arange(row.n_online)
        oracle[i] = ((row.tau_index >= 0) & (steps >= row.tau_index)).astype(float)
        constant[i] = steps.astype(float)
    assert ts_auc(oracle, meta) == pytest.approx(1.0)
    assert ts_auc(constant, meta) == pytest.approx(0.5)


def test_auc_report_slices(offline_ds):
    """The report has an overall row plus per-type slices with valid AUCs."""
    report = auc_report(baseline_scores(offline_ds)["ks"], offline_ds.meta)
    assert report.slice.iloc[0] == "overall"
    assert report.slice.str.startswith("type=").any()
    aucs = report.auc.dropna()
    assert ((aucs >= 0) & (aucs <= 1)).all()


def test_unknown_preset():
    with pytest.raises(ValueError):
        get_preset("nope")


def test_layout_defaults_are_offline():
    assert LayoutConfig().break_at_boundary is True
