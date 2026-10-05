"""Configuration and calibrated presets for structural-break data generation.

Two presets mirror the ADIA Lab structural-break competitions:

* ``adia_offline`` — 2025 edition. Raw return-scale series; the break (if any)
  sits exactly at the pre/post boundary; ~29% of series break.
* ``adia_realtime`` — 2026 edition. A break-free history followed by an online
  segment; the break (if any) occurs anywhere inside the online segment; 50% of
  series break; values are z-scored with the history's mean/std.

Process priors were calibrated against the public training sets so that the
nulls reproduce the real data's volatility clustering, tail weight and
autocorrelation spread, and the break magnitudes give simple baseline
detectors a similar AUC to the one they achieve on the real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple


@dataclass
class LayoutConfig:
    """Series lengths and break placement.

    Args:
        hist_len: Inclusive range of the pre-break / historical segment length.
        online_len: Inclusive range of the post / online segment length.
        break_prob: Probability a series contains a break.
        break_at_boundary: If True the break starts exactly at the first online
            observation (offline layout). Otherwise it is uniform in the online
            segment (real-time layout).
        zscore_by_history: Standardize each series by its history's mean/std.
    """

    hist_len: Tuple[int, int] = (1000, 2500)
    online_len: Tuple[int, int] = (250, 1000)
    break_prob: float = 0.29
    break_at_boundary: bool = True
    zscore_by_history: bool = False


@dataclass
class ProcessPrior:
    """Prior over the AR(1)-GARCH(1,1)-t base process of each series.

    The base process ``x_t`` has unit unconditional variance; the observed series
    is ``mu + sigma * x_t``.

    Args:
        sigma_median: Median per-series scale.
        sigma_log_sd: Log-normal dispersion of the per-series scale.
        mu_sd: Std of the per-series mean, in units of sigma.
        phi_sd: Std of the AR(1) coefficient for the "typical" series.
        phi_wide_prob: Fraction of series whose AR coefficient is instead drawn
            uniformly from ``phi_wide_range`` (strongly autocorrelated series).
        phi_wide_range: Range for the wide AR draws.
        garch_prob: Fraction of series with GARCH volatility clustering.
        alpha_range: Range of the ARCH coefficient for GARCH series.
        persistence_range: Range of alpha + beta for GARCH series.
        fat_tail_prob: Fraction of series with Student-t (not Gaussian) shocks.
        nu_inv_range: Range of 1/df for fat-tailed series (df >= 1/max).
        slow_vol_prob: Fraction of series with an extra slow log-volatility
            factor (a stationary AR(1)), as in component-GARCH models. Real
            "no-break" series drift in volatility over hundreds of steps; this
            reproduces the resulting false positives of variance/KS detectors.
        slow_vol_sd: Range of the slow factor's stationary std (log units).
        slow_vol_half_life: Range of the slow factor's half-life, in steps.
    """

    sigma_median: float = 0.013
    sigma_log_sd: float = 0.9
    mu_sd: float = 0.03
    phi_sd: float = 0.07
    phi_wide_prob: float = 0.03
    phi_wide_range: Tuple[float, float] = (-0.6, 0.6)
    garch_prob: float = 0.85
    alpha_range: Tuple[float, float] = (0.05, 0.15)
    persistence_range: Tuple[float, float] = (0.95, 0.995)
    fat_tail_prob: float = 0.9
    nu_inv_range: Tuple[float, float] = (0.05, 0.3)
    slow_vol_prob: float = 0.5
    slow_vol_sd: Tuple[float, float] = (0.1, 0.4)
    slow_vol_half_life: Tuple[float, float] = (250.0, 2500.0)


@dataclass
class BreakPrior:
    """Prior over break types and magnitudes.

    Magnitudes are drawn as ``N(0, sd * magnitude_scale)``; the sign is random.
    Use ``magnitude_scale`` to sweep difficulty (2.0 = breaks twice as large).

    Args:
        type_weights: Relative frequency of each break type. ``compound``
            combines two distinct single types.
        magnitude_scale: Global multiplier on every magnitude sd.
        mean_shift_sd: Mean shift, in units of the pre-break sigma.
        log_scale_sd: Change in log(sigma).
        phi_delta_sd: Change in the AR(1) coefficient.
        nu_inv_delta_sd: Change in 1/df of the shocks (tail weight).
        alpha_delta_sd: Change in the ARCH coefficient (clustering intensity).
        gradual_prob: Fraction of breaks that ramp in linearly instead of
            switching instantly.
        gradual_len: Range of the ramp length for gradual breaks.
    """

    type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "mean": 0.2,
            "variance": 0.35,
            "ar": 0.15,
            "tails": 0.1,
            "garch": 0.1,
            "compound": 0.1,
        }
    )
    magnitude_scale: float = 1.0
    mean_shift_sd: float = 0.06
    log_scale_sd: float = 0.5
    phi_delta_sd: float = 0.15
    nu_inv_delta_sd: float = 0.06
    alpha_delta_sd: float = 0.1
    gradual_prob: float = 0.2
    gradual_len: Tuple[int, int] = (10, 100)


@dataclass
class DecoyConfig:
    """Transient, non-structural events applied independently of the label.

    Decoys are "hard negatives": a single large outlier or a short volatility
    burst that decays back to the old regime. They appear in break and null
    series alike so a detector cannot use them as a label shortcut.

    Args:
        prob: Probability that a series gets a decoy.
        outlier_prob: Share of decoys that are single outliers (rest are bursts).
        outlier_size: Range of the outlier size, in base-process std units.
        burst_peak: Range of the peak volatility multiplier of a burst.
        burst_half_life: Range of the burst's decay half-life, in steps.
    """

    prob: float = 0.15
    outlier_prob: float = 0.5
    outlier_size: Tuple[float, float] = (6.0, 12.0)
    burst_peak: Tuple[float, float] = (2.0, 4.0)
    burst_half_life: Tuple[float, float] = (5.0, 30.0)


@dataclass
class GeneratorConfig:
    """Full configuration for :func:`synfin.structural_breaks.generate`."""

    layout: LayoutConfig = field(default_factory=LayoutConfig)
    process: ProcessPrior = field(default_factory=ProcessPrior)
    breaks: BreakPrior = field(default_factory=BreakPrior)
    decoys: DecoyConfig = field(default_factory=DecoyConfig)


def _adia_offline() -> GeneratorConfig:
    return GeneratorConfig()


def _adia_realtime() -> GeneratorConfig:
    return GeneratorConfig(
        layout=LayoutConfig(
            hist_len=(1000, 5000),
            online_len=(10, 999),
            break_prob=0.5,
            break_at_boundary=False,
            zscore_by_history=True,
        ),
        process=ProcessPrior(
            sigma_median=1.0,
            sigma_log_sd=0.0,
            mu_sd=0.0,
            phi_sd=0.05,
            phi_wide_prob=0.4,
            phi_wide_range=(-0.5, 0.8),
            garch_prob=0.35,
            alpha_range=(0.03, 0.12),
            persistence_range=(0.8, 0.97),
            fat_tail_prob=0.35,
            nu_inv_range=(0.05, 0.2),
            slow_vol_prob=0.25,
        ),
        breaks=BreakPrior(
            mean_shift_sd=0.08,
            log_scale_sd=0.15,
            phi_delta_sd=0.1,
            nu_inv_delta_sd=0.06,
            alpha_delta_sd=0.06,
        ),
    )


PRESETS = {
    "adia_offline": _adia_offline,
    "adia_realtime": _adia_realtime,
}


def get_preset(name: str, magnitude_scale: float = 1.0) -> GeneratorConfig:
    """Return a fresh copy of a named preset.

    Args:
        name: One of :data:`PRESETS`.
        magnitude_scale: Overrides ``breaks.magnitude_scale``.

    Returns:
        A new :class:`GeneratorConfig`.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose from {sorted(PRESETS)}.")
    cfg = PRESETS[name]()
    cfg.breaks = replace(cfg.breaks, magnitude_scale=magnitude_scale)
    return cfg
