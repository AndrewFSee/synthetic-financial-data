"""Side-by-side distribution comparison plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_distributions(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: Optional[List[str]] = None,
    bins: int = 50,
    figsize: Optional[tuple] = None,
    savepath: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Plot side-by-side histograms/KDE comparing real vs synthetic distributions.

    Args:
        real: Real data, shape (N, num_features) or (N, seq_len, num_features).
        synthetic: Synthetic data, same shape.
        feature_names: Feature names for subplot titles.
        bins: Number of histogram bins.
        figsize: Figure size (width, height). Auto-computed if None.
        savepath: Optional path to save the figure.
        show: Whether to display the figure.
    """
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])
    if synthetic.ndim == 3:
        synthetic = synthetic.reshape(-1, synthetic.shape[-1])

    num_features = real.shape[1] if real.ndim > 1 else 1
    if real.ndim == 1:
        real = real[:, None]
        synthetic = synthetic[:, None]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    ncols = min(4, num_features)
    nrows = (num_features + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten() if num_features > 1 else [axes]

    for i in range(num_features):
        ax = axes[i]
        ax.hist(real[:, i], bins=bins, alpha=0.5, label="Real", density=True, color="steelblue")
        ax.hist(synthetic[:, i], bins=bins, alpha=0.5, label="Synthetic", density=True, color="coral")
        ax.set_title(feature_names[i])
        ax.legend(fontsize=8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info("Saved distribution plot to %s", savepath)
    if show:
        plt.show()
    plt.close(fig)
