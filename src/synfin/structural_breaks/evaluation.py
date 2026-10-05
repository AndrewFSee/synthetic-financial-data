"""Score break detectors against generated ground truth.

* :func:`auc_report` — series-level ROC AUC, overall and sliced by break type,
  effect-size bin and decoy type. Slices are where synthetic data beats the
  real competition data: you can see *which* kinds of breaks a model misses.
* :func:`ts_auc` — the real-time competition's time-stratified AUC.
* :func:`baseline_scores` — simple two-sample statistics (history vs online)
  that serve as a difficulty yardstick for a dataset.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score

from synfin.structural_breaks.generator import StructuralBreakDataset


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[1:], x[:-1]) / denom) if denom > 0 else 0.0


def baseline_scores(ds: StructuralBreakDataset) -> pd.DataFrame:
    """Two-sample statistics comparing each series' history to its online part.

    Args:
        ds: Generated dataset.

    Returns:
        DataFrame indexed by id with one column per baseline detector; larger
        values mean "more likely a break".
    """
    rows = []
    for s, n_hist in zip(ds.series, ds.meta["n_hist"].to_numpy()):
        pre, post = s[:n_hist], s[n_hist:]
        v_pre, v_post = pre.var() + 1e-24, post.var() + 1e-24
        rows.append(
            {
                "log_var_ratio": abs(np.log(v_post / v_pre)),
                "mean_t": abs(post.mean() - pre.mean())
                / np.sqrt(v_pre / len(pre) + v_post / len(post)),
                "ks": ks_2samp(pre, post).statistic if len(post) > 1 else 0.0,
                "acf1_diff": abs(_acf1(post) - _acf1(pre)) if len(post) > 2 else 0.0,
                "abs_acf1_diff": (
                    abs(_acf1(np.abs(post)) - _acf1(np.abs(pre))) if len(post) > 2 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows, index=ds.meta.index)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auc_report(
    scores: pd.Series,
    meta: pd.DataFrame,
    n_effect_bins: int = 3,
) -> pd.DataFrame:
    """Series-level ROC AUC, overall and per slice.

    Each break slice compares the breaks in that slice against *all* nulls, so
    slice AUCs are directly comparable ("how well are mean breaks separated
    from no-break series?").

    Args:
        scores: Detector score per series, indexed like ``meta``.
        meta: The dataset's ``meta`` frame.
        n_effect_bins: Quantile bins of effect size within each single type.

    Returns:
        DataFrame with columns ``slice``, ``n_breaks``, ``auc``.
    """
    scores = scores.reindex(meta.index)
    y = meta["has_break"].to_numpy()
    s = scores.to_numpy(dtype=float)
    null = ~y
    rows = [{"slice": "overall", "n_breaks": int(y.sum()), "auc": _auc(y, s)}]

    def add(name: str, mask: np.ndarray) -> None:
        sel = null | mask
        rows.append({"slice": name, "n_breaks": int(mask.sum()), "auc": _auc(y[sel], s[sel])})

    types = meta["break_type"].to_numpy()
    for t in sorted({t for t in types[y] if "+" not in t}):
        mask = types == t
        add(f"type={t}", mask)
        if mask.sum() < n_effect_bins * 5:
            continue
        bins = pd.qcut(meta.loc[mask, "effect"], n_effect_bins, duplicates="drop")
        for interval in bins.cat.categories:
            in_bin = np.zeros(len(meta), dtype=bool)
            in_bin[np.flatnonzero(mask)[(bins == interval).to_numpy()]] = True
            lo = max(interval.left, 0.0)
            add(f"type={t} effect={lo:.3g}..{interval.right:.3g}", in_bin)
    compound = np.char.find(types.astype(str), "+") >= 0
    if compound.any():
        add("type=compound", compound)

    if "ramp_len" in meta:
        add("gradual breaks", y & (meta["ramp_len"] > 1).to_numpy())
    for d in ("outlier", "vol_burst"):
        decoy = (meta["decoy"] == d).to_numpy()
        if decoy.any():
            # Decoy nulls vs all breaks: does the transient fool the detector?
            sel = y | (null & decoy)
            rows.append(
                {
                    "slice": f"decoy={d} (decoy nulls vs breaks)",
                    "n_breaks": int(y.sum()),
                    "auc": _auc(y[sel], s[sel]),
                }
            )
    return pd.DataFrame(rows)


def ts_auc(step_scores: Mapping[int, Sequence[float]], meta: pd.DataFrame) -> float:
    """Time-stratified AUC used by the real-time competition.

    At each online step ``t`` the series still alive are labeled positive if
    their break has occurred (``t >= tau_index``). Per-step AUCs are averaged
    with weights ``n_pos(t) * n_neg(t)``.

    Args:
        step_scores: Map from series id to its per-online-step scores.
        meta: The dataset's ``meta`` frame.

    Returns:
        The weighted TS-AUC.
    """
    ids = list(step_scores)
    lengths = np.array([len(step_scores[i]) for i in ids])
    max_len = int(lengths.max())
    S = np.full((len(ids), max_len), np.nan)
    for row, i in enumerate(ids):
        S[row, : lengths[row]] = step_scores[i]
    tau_index = meta.loc[ids, "tau_index"].to_numpy()

    num = den = 0.0
    for t in range(max_len):
        alive = lengths > t
        pos = alive & (tau_index >= 0) & (t >= tau_index)
        n_pos, n_neg = int(pos.sum()), int(alive.sum() - pos.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(S[alive, t])
        auc = (ranks[pos[alive]].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        w = n_pos * n_neg
        num += w * auc
        den += w
    return num / den if den else float("nan")


def summarize(ds: StructuralBreakDataset) -> Dict[str, float]:
    """Headline stylized-fact statistics of the history segments (medians).

    Useful for checking a preset against real data.
    """
    kurt, acf1, abs_acf1, abs_acf20 = [], [], [], []
    for s, n in zip(ds.series, ds.meta["n_hist"].to_numpy()):
        h = s[:n]
        a = np.abs(h) - np.abs(h).mean()
        kurt.append(pd.Series(h).kurt())
        acf1.append(_acf1(h))
        abs_acf1.append(_acf1(np.abs(h)))
        abs_acf20.append(float(np.dot(a[20:], a[:-20]) / np.dot(a, a)))
    return {
        "break_rate": float(ds.meta["has_break"].mean()),
        "excess_kurtosis": float(np.median(kurt)),
        "acf1": float(np.median(acf1)),
        "acf1_p10": float(np.percentile(acf1, 10)),
        "acf1_p90": float(np.percentile(acf1, 90)),
        "abs_acf1": float(np.median(abs_acf1)),
        "abs_acf20": float(np.median(abs_acf20)),
    }
