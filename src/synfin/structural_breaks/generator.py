"""Simulate univariate return series with labeled structural breaks.

Each series is ``r_t = mu_t + sigma_t * x_t`` where ``x_t`` is a unit-variance
AR(1)-GARCH(1,1) process with standardized Student-t shocks::

    x_t   = phi_t * x_{t-1} + eps_t
    eps_t = sqrt(h_t) * z_t,             z_t ~ t(df_t) scaled to unit variance
    h_t   = omega_t + alpha_t * eps_{t-1}^2 + beta_t * h_{t-1}
    omega_t = (1 - phi_t^2) * (1 - alpha_t - beta_t)   # keeps Var(x) = 1

optionally multiplied by a slow, stationary log-volatility factor (component-
GARCH style) so that no-break series drift in volatility like real ones do.

A break at ``tau`` moves one (or, for ``compound``, two) parameter groups from
their pre-break to post-break values, either instantly or along a linear ramp:

* ``mean``     — mu shifts (in units of the pre-break sigma)
* ``variance`` — sigma is rescaled
* ``ar``       — phi changes (variance-preserving)
* ``tails``    — the shock degrees of freedom change (variance-preserving)
* ``garch``    — the ARCH coefficient, i.e. clustering intensity, changes

Because the conditional variance carries over the break, dynamics breaks are
realistic rather than "two independent series glued together". Null series
share the same volatility clustering, so variance-style detectors face a
realistic false-positive rate.

Alternatively the base process can be a stationary block bootstrap of a real
return series (``bootstrap_source``), in which case only ``mean`` and
``variance`` breaks are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from synfin.structural_breaks.presets import GeneratorConfig

logger = logging.getLogger(__name__)

SINGLE_BREAK_TYPES = ("mean", "variance", "ar", "tails", "garch")
BOOTSTRAP_BREAK_TYPES = ("mean", "variance")
_PARAMS = ("mu", "log_sigma", "phi", "alpha", "persistence", "nu_inv")
_BURN_IN = 300
_PHI_BOUNDS = (-0.95, 0.98)
_NU_INV_BOUNDS = (0.0, 0.3)
_ALPHA_BOUNDS = (0.0, 0.3)
_MAX_PERSISTENCE = 0.995


@dataclass
class StructuralBreakDataset:
    """Generated series plus per-series ground truth.

    Attributes:
        series: One float64 array per series (history followed by online part).
        meta: One row per series, indexed by ``id``. Key columns:
            ``n_hist``, ``n_online``, ``has_break``, ``tau`` (absolute index of
            the first post-break observation, -1 if none), ``tau_index``
            (``tau - n_hist``), ``break_type``, ``effect`` (absolute size of the
            change in the type's native units; NaN for compound breaks),
            ``ramp_len``, ``decoy``, ``decoy_pos`` and the ``pre_*`` / ``post_*``
            process parameters.
    """

    series: List[np.ndarray]
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.series)


def generate(
    n_series: int,
    config: GeneratorConfig,
    seed: int = 0,
    bootstrap_source: Optional[np.ndarray] = None,
    bootstrap_block: float = 50.0,
    batch_size: int = 1000,
) -> StructuralBreakDataset:
    """Generate a labeled structural-break dataset.

    Args:
        n_series: Number of series.
        config: Generator configuration (see :func:`get_preset`).
        seed: Random seed; output is fully deterministic given the seed.
        bootstrap_source: Optional 1-D array of real returns. If given, the
            base process is a stationary block bootstrap of it instead of the
            parametric AR-GARCH-t model.
        bootstrap_block: Mean block length for the stationary bootstrap.
        batch_size: Series simulated together (memory ~ batch * max_len * 8B).

    Returns:
        A :class:`StructuralBreakDataset`.
    """
    rng = np.random.default_rng(seed)
    layout = config.layout

    n_hist = rng.integers(layout.hist_len[0], layout.hist_len[1] + 1, size=n_series)
    n_online = rng.integers(layout.online_len[0], layout.online_len[1] + 1, size=n_series)
    has_break = rng.random(n_series) < layout.break_prob
    if layout.break_at_boundary:
        tau_index = np.zeros(n_series, dtype=np.int64)
    else:
        tau_index = np.floor(rng.random(n_series) * n_online).astype(np.int64)
    tau_index = np.where(has_break, tau_index, -1)
    tau = np.where(has_break, n_hist + tau_index, -1)

    pre = _sample_process_params(n_series, config, rng)
    allowed = BOOTSTRAP_BREAK_TYPES if bootstrap_source is not None else SINGLE_BREAK_TYPES
    post, break_type, effect = _sample_breaks(pre, has_break, config, allowed, rng)

    gradual = has_break & (rng.random(n_series) < config.breaks.gradual_prob)
    lo, hi = config.breaks.gradual_len
    ramp_len = np.where(gradual, rng.integers(lo, hi + 1, size=n_series), 1)

    decoy, decoy_pos, decoy_size, decoy_hl = _sample_decoys(n_hist, n_online, config, rng)

    lengths = n_hist + n_online
    series: List[np.ndarray] = []
    for start in range(0, n_series, batch_size):
        sl = slice(start, min(start + batch_size, n_series))
        if bootstrap_source is None:
            x = _simulate_garch_batch(
                {k: v[sl] for k, v in pre.items()},
                {k: v[sl] for k, v in post.items()},
                tau[sl],
                ramp_len[sl],
                int(lengths[sl].max()),
                rng,
            )
        else:
            x = _bootstrap_batch(
                bootstrap_source, len(lengths[sl]), int(lengths[sl].max()), bootstrap_block, rng
            )
        for j, i in enumerate(range(sl.start, sl.stop)):
            xi = x[j, : lengths[i]]
            xi = _apply_decoy(xi, decoy[i], decoy_pos[i], decoy_size[i], decoy_hl[i])
            r = _to_observed(xi, pre, post, i, tau[i], ramp_len[i])
            if layout.zscore_by_history:
                hist = r[: n_hist[i]]
                r = (r - hist.mean()) / (hist.std() + 1e-12)
            series.append(r)
        logger.debug("Simulated %d / %d series", sl.stop, n_series)

    meta = pd.DataFrame(
        {
            "n_hist": n_hist,
            "n_online": n_online,
            "has_break": has_break,
            "tau": tau,
            "tau_index": tau_index,
            "break_type": break_type,
            "effect": effect,
            "ramp_len": ramp_len,
            "decoy": decoy,
            "decoy_pos": decoy_pos,
        },
        index=pd.RangeIndex(n_series, name="id"),
    )
    for k in _PARAMS:
        meta[f"pre_{k}"] = pre[k]
        meta[f"post_{k}"] = post[k]
    meta["slow_vol_sd"] = pre["slow_vol_sd"]
    meta["slow_vol_half_life"] = pre["slow_vol_half_life"]
    if bootstrap_source is not None:
        unused = [f"{p}_{k}" for p in ("pre", "post") for k in _PARAMS[2:]]
        meta[unused + ["slow_vol_sd", "slow_vol_half_life"]] = np.nan
    return StructuralBreakDataset(series=series, meta=meta)


# ----------------------------------------------------------------------
# Parameter sampling
# ----------------------------------------------------------------------


def _sample_process_params(
    n: int, config: GeneratorConfig, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Draw pre-break base-process parameters for every series."""
    p = config.process
    log_sigma = np.log(p.sigma_median) + p.sigma_log_sd * rng.standard_normal(n)
    mu = p.mu_sd * np.exp(log_sigma) * rng.standard_normal(n)

    phi = p.phi_sd * rng.standard_normal(n)
    wide = rng.random(n) < p.phi_wide_prob
    phi[wide] = rng.uniform(*p.phi_wide_range, size=wide.sum())
    phi = np.clip(phi, *_PHI_BOUNDS)

    garch = rng.random(n) < p.garch_prob
    alpha = np.where(garch, rng.uniform(*p.alpha_range, size=n), 0.0)
    persistence = np.where(garch, rng.uniform(*p.persistence_range, size=n), 0.0)
    persistence = np.maximum(persistence, alpha)

    fat = rng.random(n) < p.fat_tail_prob
    nu_inv = np.where(fat, rng.uniform(*p.nu_inv_range, size=n), 0.0)

    slow = rng.random(n) < p.slow_vol_prob
    slow_sd = np.where(slow, rng.uniform(*p.slow_vol_sd, size=n), 0.0)
    slow_half_life = np.where(slow, rng.uniform(*p.slow_vol_half_life, size=n), 0.0)

    return {
        "mu": mu,
        "log_sigma": log_sigma,
        "phi": phi,
        "alpha": alpha,
        "persistence": persistence,
        "nu_inv": nu_inv,
        "slow_vol_sd": slow_sd,
        "slow_vol_half_life": slow_half_life,
    }


def _reflect(value: np.ndarray, delta: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Apply ``value + delta``, flipping the sign of delta if it leaves [lo, hi]."""
    out = value + delta
    flip = (out < lo) | (out > hi)
    out[flip] = value[flip] - delta[flip]
    return np.clip(out, lo, hi)


def _sample_breaks(
    pre: Dict[str, np.ndarray],
    has_break: np.ndarray,
    config: GeneratorConfig,
    allowed: tuple,
    rng: np.random.Generator,
):
    """Draw post-break parameters, break types and effect sizes."""
    b = config.breaks
    n = len(has_break)
    weights = {k: v for k, v in b.type_weights.items() if k in allowed or k == "compound"}
    names = list(weights)
    probs = np.array([weights[k] for k in names], dtype=float)
    probs /= probs.sum()

    break_type = np.full(n, "none", dtype=object)
    break_type[has_break] = rng.choice(names, size=has_break.sum(), p=probs)
    effect = np.zeros(n)
    post = {k: v.copy() for k, v in pre.items()}

    # Resolve each break into the set of single types it applies.
    components: Dict[str, np.ndarray] = {t: break_type == t for t in allowed}
    compound = np.flatnonzero(break_type == "compound")
    for i in compound:
        pair = rng.choice(allowed, size=2, replace=False)
        break_type[i] = "+".join(sorted(pair))
        for t in pair:
            components[t][i] = True
        effect[i] = np.nan

    s = b.magnitude_scale
    for t, mask in components.items():
        m = int(mask.sum())
        if m == 0:
            continue
        idx = np.flatnonzero(mask)
        if t == "mean":
            d = b.mean_shift_sd * s * rng.standard_normal(m)
            post["mu"][idx] = pre["mu"][idx] + d * np.exp(pre["log_sigma"][idx])
        elif t == "variance":
            d = b.log_scale_sd * s * rng.standard_normal(m)
            post["log_sigma"][idx] = pre["log_sigma"][idx] + d
        elif t == "ar":
            d = b.phi_delta_sd * s * rng.standard_normal(m)
            post["phi"][idx] = _reflect(pre["phi"][idx], d, *_PHI_BOUNDS)
            d = post["phi"][idx] - pre["phi"][idx]
        elif t == "tails":
            d = b.nu_inv_delta_sd * s * rng.standard_normal(m)
            post["nu_inv"][idx] = _reflect(pre["nu_inv"][idx], d, *_NU_INV_BOUNDS)
            d = post["nu_inv"][idx] - pre["nu_inv"][idx]
        elif t == "garch":
            d = b.alpha_delta_sd * s * rng.standard_normal(m)
            new_alpha = _reflect(pre["alpha"][idx], d, *_ALPHA_BOUNDS)
            post["alpha"][idx] = new_alpha
            # A previously non-GARCH series needs a persistence to cluster at all.
            base = np.where(pre["persistence"][idx] > 0, pre["persistence"][idx], 0.9)
            post["persistence"][idx] = np.clip(base, new_alpha + 0.01, _MAX_PERSISTENCE)
            d = new_alpha - pre["alpha"][idx]
        single = idx[break_type[idx] == t]
        effect[single] = np.abs(d[np.isin(idx, single)])

    return post, break_type.astype(str), effect


def _sample_decoys(
    n_hist: np.ndarray, n_online: np.ndarray, config: GeneratorConfig, rng: np.random.Generator
):
    """Draw transient decoy events (independent of the break label)."""
    d = config.decoys
    n = len(n_hist)
    has = rng.random(n) < d.prob
    is_outlier = rng.random(n) < d.outlier_prob
    decoy = np.where(has, np.where(is_outlier, "outlier", "vol_burst"), "none")
    pos = np.where(has, n_hist + np.floor(rng.random(n) * n_online).astype(np.int64), -1)
    size = np.where(
        is_outlier,
        rng.uniform(*d.outlier_size, size=n) * rng.choice([-1.0, 1.0], size=n),
        rng.uniform(*d.burst_peak, size=n),
    )
    half_life = rng.uniform(*d.burst_half_life, size=n)
    return decoy, pos, size, half_life


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------


def _ramp_weights(tau: np.ndarray, ramp_len: np.ndarray, length: int) -> np.ndarray:
    """Weight in [0, 1] of the post-break regime at each step, shape (N, length)."""
    t = np.arange(length)[None, :]
    tau_eff = np.where(tau >= 0, tau, np.iinfo(np.int64).max // 2)[:, None]
    return np.clip((t - tau_eff + 1) / ramp_len[:, None], 0.0, 1.0)


def _simulate_garch_batch(
    pre: Dict[str, np.ndarray],
    post: Dict[str, np.ndarray],
    tau: np.ndarray,
    ramp_len: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the unit-variance AR-GARCH-t base process for a batch of series."""
    n = len(tau)
    w_all = _ramp_weights(tau, ramp_len, length)
    dyn = ("phi", "alpha", "persistence", "nu_inv")
    changes = {k: post[k] - pre[k] for k in dyn}
    any_dyn_change = np.zeros(n, dtype=bool)
    for k in dyn:
        any_dyn_change |= changes[k] != 0

    def params_at(w: np.ndarray):
        phi = pre["phi"] + w * changes["phi"]
        alpha = pre["alpha"] + w * changes["alpha"]
        pers = pre["persistence"] + w * changes["persistence"]
        nu_inv = pre["nu_inv"] + w * changes["nu_inv"]
        omega = (1.0 - phi**2) * (1.0 - pers)
        df = np.where(nu_inv > 1e-4, 1.0 / np.maximum(nu_inv, 1e-4), 1e8)
        return phi, alpha, pers - alpha, omega, df

    out = np.empty((n, length))
    zero = np.zeros(n)
    phi, alpha, beta, omega, df = params_at(zero)
    h = 1.0 - phi**2  # stationary Var(eps)
    eps_prev = np.sqrt(h) * rng.standard_normal(n)
    x_prev = rng.standard_normal(n)

    # Slow log-volatility factor, started from its stationary distribution.
    slow_sd = pre["slow_vol_sd"]
    has_slow = slow_sd > 0
    rho = np.where(has_slow, 0.5 ** (1.0 / np.maximum(pre["slow_vol_half_life"], 1.0)), 0.0)
    innov_sd = slow_sd * np.sqrt(1.0 - rho**2)
    log_vol = slow_sd * rng.standard_normal(n)
    vol_norm = np.exp(-(slow_sd**2))  # E[exp(2 * log_vol)] = exp(2 sd^2)

    for step in range(-_BURN_IN, length):
        if step >= 0 and any_dyn_change.any():
            phi, alpha, beta, omega, df = params_at(w_all[:, step])
        h = omega + alpha * eps_prev**2 + beta * h
        g = rng.standard_gamma(df / 2.0) / (df / 2.0)
        z = rng.standard_normal(n) / np.sqrt(g) * np.sqrt((df - 2.0) / df)
        eps_prev = np.sqrt(h) * z
        x_prev = phi * x_prev + eps_prev
        if step >= 0:
            log_vol = rho * log_vol + innov_sd * rng.standard_normal(n)
            out[:, step] = x_prev * np.exp(log_vol) * vol_norm
    return out


def _bootstrap_batch(
    source: np.ndarray, n: int, length: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Stationary (Politis-Romano) block bootstrap of a standardized source series."""
    src = np.asarray(source, dtype=float)
    src = src[np.isfinite(src)]
    if len(src) < 2 * mean_block:
        raise ValueError(f"bootstrap_source too short ({len(src)}) for block {mean_block}.")
    src = (src - src.mean()) / src.std()
    m = len(src)
    out = np.empty((n, length))
    idx = rng.integers(0, m, size=n)
    p_new = 1.0 / mean_block
    for t in range(length):
        out[:, t] = src[idx]
        jump = rng.random(n) < p_new
        idx = np.where(jump, rng.integers(0, m, size=n), (idx + 1) % m)
    return out


def _apply_decoy(x: np.ndarray, kind: str, pos: int, size: float, half_life: float) -> np.ndarray:
    """Inject a transient outlier or decaying volatility burst at ``pos``."""
    if kind == "none":
        return x
    x = x.copy()
    if kind == "outlier":
        x[pos] += size
    else:
        k = np.arange(len(x) - pos)
        x[pos:] *= 1.0 + (size - 1.0) * np.exp(-np.log(2.0) * k / half_life)
    return x


def _to_observed(
    x: np.ndarray,
    pre: Dict[str, np.ndarray],
    post: Dict[str, np.ndarray],
    i: int,
    tau: int,
    ramp_len: int,
) -> np.ndarray:
    """Map the unit base process to the observed scale: mu_t + sigma_t * x_t."""
    w = _ramp_weights(np.array([tau]), np.array([ramp_len]), len(x))[0]
    mu = pre["mu"][i] + w * (post["mu"][i] - pre["mu"][i])
    log_sigma = pre["log_sigma"][i] + w * (post["log_sigma"][i] - pre["log_sigma"][i])
    return mu + np.exp(log_sigma) * x
