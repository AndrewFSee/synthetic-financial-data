"""Export generated datasets in the ADIA Lab structural-break file formats.

Offline (2025) layout::

    X_train.parquet  MultiIndex (id, time); columns value (float64), period (0 pre / 1 post)
    y_train.parquet  index id; column structural_breakpoint (bool)

Real-time (2026) layout::

    X_train.parquet        MultiIndex (id, time); columns value, period (1 history / 2 online)
    y_train_index.parquet  index id; columns tau_index, tau (-1 if no break)
    y_train.parquet        MultiIndex (id, time) over online rows; column target = 1[time >= tau]

Both layouts also write ``meta_train.parquet`` with the full ground truth
(break type, effect size, decoys, process parameters) for sliced evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from synfin.structural_breaks.generator import StructuralBreakDataset

logger = logging.getLogger(__name__)


def _long_frame(ds: StructuralBreakDataset, pre_period: int, post_period: int) -> pd.DataFrame:
    lengths = np.array([len(s) for s in ds.series])
    ids = np.repeat(ds.meta.index.to_numpy(), lengths)
    time = np.concatenate([np.arange(n) for n in lengths])
    n_hist = np.repeat(ds.meta["n_hist"].to_numpy(), lengths)
    period = np.where(time < n_hist, pre_period, post_period).astype(np.int64)
    index = pd.MultiIndex.from_arrays([ids, time], names=["id", "time"])
    return pd.DataFrame({"value": np.concatenate(ds.series), "period": period}, index=index)


def to_adia_offline(ds: StructuralBreakDataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert to the 2025 offline format.

    Only meaningful for datasets whose breaks sit at the boundary
    (``LayoutConfig.break_at_boundary=True``).

    Returns:
        Tuple ``(X, y)``.
    """
    if (ds.meta.loc[ds.meta["has_break"], "tau_index"] != 0).any():
        raise ValueError("Offline format requires every break to sit at the period boundary.")
    X = _long_frame(ds, pre_period=0, post_period=1)
    y = ds.meta[["has_break"]].rename(columns={"has_break": "structural_breakpoint"})
    return X, y


def to_adia_realtime(
    ds: StructuralBreakDataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert to the 2026 real-time format.

    Returns:
        Tuple ``(X, y_index, y_steps)``.
    """
    X = _long_frame(ds, pre_period=1, post_period=2)
    y_index = ds.meta[["tau_index", "tau"]].astype(np.int64)

    online = X[X["period"] == 2]
    ids = online.index.get_level_values("id").to_numpy()
    time = online.index.get_level_values("time").to_numpy()
    tau = ds.meta["tau"].to_numpy()[ids]
    target = ((tau >= 0) & (time >= tau)).astype(np.int64)
    y_steps = pd.DataFrame({"target": target}, index=online.index)
    return X, y_index, y_steps


def save_adia(
    ds: StructuralBreakDataset,
    output_dir: Union[str, Path],
    layout: str = "offline",
    split: str = "train",
) -> Path:
    """Write a dataset to ``output_dir`` in an ADIA layout.

    Args:
        ds: Generated dataset.
        output_dir: Destination directory (created if missing).
        layout: ``"offline"`` or ``"realtime"``.
        split: File-name suffix, e.g. ``"train"`` or ``"test"``.

    Returns:
        The output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if layout == "offline":
        X, y = to_adia_offline(ds)
        y.to_parquet(out / f"y_{split}.parquet")
    elif layout == "realtime":
        X, y_index, y_steps = to_adia_realtime(ds)
        y_index.to_parquet(out / f"y_{split}_index.parquet")
        y_steps.to_parquet(out / f"y_{split}.parquet")
    else:
        raise ValueError(f"Unknown layout {layout!r}. Use 'offline' or 'realtime'.")
    X.to_parquet(out / f"X_{split}.parquet")
    ds.meta.to_parquet(out / f"meta_{split}.parquet")
    logger.info("Wrote %d series (%s layout) to %s", len(ds), layout, out)
    return out
