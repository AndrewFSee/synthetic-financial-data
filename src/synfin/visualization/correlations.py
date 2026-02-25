"""Cross-correlation heatmap visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (14, 5),
    savepath: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Plot real vs synthetic correlation matrices as heatmaps.

    Args:
        real: Real data, shape (N, num_features) or (N, seq_len, num_features).
        synthetic: Synthetic data, same shape.
        feature_names: Feature names for axis labels.
        figsize: Figure size.
        savepath: Optional path to save the figure.
        show: Whether to display the figure.
    """
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])
    if synthetic.ndim == 3:
        synthetic = synthetic.reshape(-1, synthetic.shape[-1])

    n_features = real.shape[1]
    labels = feature_names or [f"F{i}" for i in range(n_features)]

    real_corr = np.corrcoef(real, rowvar=False)
    synth_corr = np.corrcoef(synthetic, rowvar=False)
    diff = real_corr - synth_corr

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    try:
        import seaborn as sns
        for ax, data, title in [
            (axes[0], real_corr, "Real Correlations"),
            (axes[1], synth_corr, "Synthetic Correlations"),
            (axes[2], diff, "Difference (Real - Synthetic)"),
        ]:
            cmap = "RdBu_r" if "Diff" in title else "coolwarm"
            sns.heatmap(
                data,
                ax=ax,
                cmap=cmap,
                center=0,
                annot=n_features <= 10,
                fmt=".2f",
                xticklabels=labels,
                yticklabels=labels,
            )
            ax.set_title(title)
    except ImportError:
        for ax, data, title in [
            (axes[0], real_corr, "Real Correlations"),
            (axes[1], synth_corr, "Synthetic Correlations"),
            (axes[2], diff, "Difference"),
        ]:
            im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info("Saved correlation heatmap to %s", savepath)
    if show:
        plt.show()
    plt.close(fig)
