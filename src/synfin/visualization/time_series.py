"""Multi-panel time series plots: price, volume, and indicators."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_time_series(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[List[str]] = None,
    num_samples: int = 3,
    figsize: Optional[tuple] = None,
    savepath: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Plot side-by-side real vs synthetic time series.

    Args:
        real: Real sequences, shape (N, seq_len, num_features).
        synthetic: Synthetic sequences, shape (M, seq_len, num_features).
        feature_names: Feature names for panel labels.
        num_samples: Number of sample sequences to plot.
        figsize: Figure size. Auto-computed if None.
        savepath: Optional path to save the figure.
        show: Whether to display the figure.
    """
    n_features = real.shape[-1] if real.ndim == 3 else 1
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    num_samples = min(num_samples, len(real), len(synthetic))
    if figsize is None:
        figsize = (14, 3 * min(n_features, 5))

    fig, axes = plt.subplots(
        min(n_features, 5), 2,
        figsize=figsize,
        sharex=True,
    )
    if min(n_features, 5) == 1:
        axes = axes[None, :]

    for feat_idx in range(min(n_features, 5)):
        ax_real = axes[feat_idx, 0]
        ax_synth = axes[feat_idx, 1]

        for i in range(num_samples):
            r = real[i, :, feat_idx] if real.ndim == 3 else real[:, feat_idx]
            s = synthetic[i, :, feat_idx] if synthetic.ndim == 3 else synthetic[:, feat_idx]
            ax_real.plot(r, alpha=0.7, linewidth=0.8)
            ax_synth.plot(s, alpha=0.7, linewidth=0.8)

        ax_real.set_ylabel(feature_names[feat_idx], fontsize=9)
        if feat_idx == 0:
            ax_real.set_title("Real", fontsize=11)
            ax_synth.set_title("Synthetic", fontsize=11)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info("Saved time series plot to %s", savepath)
    if show:
        plt.show()
    plt.close(fig)
