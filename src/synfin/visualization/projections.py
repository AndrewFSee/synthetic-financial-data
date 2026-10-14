"""PCA and t-SNE projections of real vs synthetic windows (TimeGAN-paper style).

Each window is reduced to one point and projected to 2-D. Overlapping clouds
mean the synthetic windows cover the same region as the real ones; a synthetic
cloud that sits inside a small part of the real one signals mode collapse, and
separate clouds signal a distribution mismatch.

``reduce="mean_time"`` (default, as in the TimeGAN paper and common notebooks)
averages each window over time: one point per window, F dimensions. That keeps
each window's level per feature but discards its temporal structure.
``reduce="flatten"`` keeps every time step (T x F dimensions). Features are
standardized with the real data's statistics first, because they have
different units.

These plots are a visual sanity check, not a test: the quantitative versions
are the discriminative scores in :mod:`synfin.evaluation`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def compute_projections(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_samples: int = 1000,
    reduce: str = "mean_time",
    perplexity: float = 40.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """2-D PCA (fit on real) and joint t-SNE embeddings of real and synthetic windows.

    Args:
        real: Real windows (N, T, F).
        synthetic: Synthetic windows (M, T, F).
        n_samples: Windows drawn from each set (fewer if not available).
        reduce: "mean_time" (average over time) or "flatten" (all steps).
        perplexity: t-SNE perplexity (capped for small samples).
        seed: Random seed for subsampling and t-SNE.

    Returns:
        Dict with ``real_pca``, ``synthetic_pca``, ``real_tsne``,
        ``synthetic_tsne`` (each (n, 2)) and ``pca_explained`` (variance ratio).
    """
    if reduce not in ("mean_time", "flatten"):
        raise ValueError(f"Unknown reduce {reduce!r}; use 'mean_time' or 'flatten'.")
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(real), len(synthetic))
    r = real[rng.choice(len(real), n, replace=False)]
    s = synthetic[rng.choice(len(synthetic), n, replace=False)]

    flat_r = real.reshape(-1, real.shape[-1])
    mu, sd = flat_r.mean(axis=0), flat_r.std(axis=0) + 1e-12
    r, s = (r - mu) / sd, (s - mu) / sd
    if reduce == "mean_time":
        r, s = r.mean(axis=1), s.mean(axis=1)
    else:
        r, s = r.reshape(n, -1), s.reshape(n, -1)

    pca = PCA(n_components=2).fit(r)
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(2.0, (2 * n - 1) / 3.0)),
        init="pca",
        random_state=seed,
    )
    emb = tsne.fit_transform(np.vstack([r, s]))
    return {
        "real_pca": pca.transform(r),
        "synthetic_pca": pca.transform(s),
        "real_tsne": emb[:n],
        "synthetic_tsne": emb[n:],
        "pca_explained": pca.explained_variance_ratio_,
    }


def plot_projections(
    real: np.ndarray,
    synthetic: np.ndarray,
    title: Optional[str] = None,
    savepath: Optional[Union[str, Path]] = None,
    show: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Side-by-side PCA and t-SNE scatter plots of real vs synthetic windows.

    Args:
        real: Real windows (N, T, F).
        synthetic: Synthetic windows (M, T, F).
        title: Optional figure title.
        savepath: Optional path to save the figure.
        show: Whether to display the figure.
        **kwargs: Passed to :func:`compute_projections`.

    Returns:
        The projections from :func:`compute_projections`.
    """
    p = compute_projections(real, synthetic, **kwargs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    style = {"s": 8, "alpha": 0.45, "linewidths": 0}
    for ax, key, name in [(axes[0], "pca", "PCA"), (axes[1], "tsne", "t-SNE")]:
        ax.scatter(*p[f"real_{key}"].T, c="#1f77b4", label="real", **style)
        ax.scatter(*p[f"synthetic_{key}"].T, c="#d62728", label="synthetic", **style)
        sub = ""
        if key == "pca":
            sub = " (fit on real; {:.0%} + {:.0%} of variance)".format(*p["pca_explained"])
        ax.set_title(name + sub, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, frameon=False, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        logger.info("Saved projection plot to %s", savepath)
    if show:
        plt.show()
    plt.close(fig)
    return p
