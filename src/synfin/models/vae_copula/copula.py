"""Copula dependency modeling: Gaussian and Student-t copulas."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy import stats
from torch import Tensor

logger = logging.getLogger(__name__)


class GaussianCopula:
    """Gaussian copula for modeling cross-feature dependencies.

    Workflow:
      1. fit(z): Fit the copula to latent representations.
      2. sample(n): Sample correlated latent vectors.
      3. Decode sampled vectors to get synthetic sequences.

    Args:
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.corr_matrix: Optional[np.ndarray] = None
        self.marginal_means: Optional[np.ndarray] = None
        self.marginal_stds: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, z: np.ndarray) -> "GaussianCopula":
        """Fit the Gaussian copula to latent representations.

        Args:
            z: Latent vectors, shape (N, latent_dim).

        Returns:
            self (for chaining).
        """
        self.marginal_means = z.mean(axis=0)
        self.marginal_stds = z.std(axis=0) + 1e-8
        z_norm = (z - self.marginal_means) / self.marginal_stds
        self.corr_matrix = np.corrcoef(z_norm, rowvar=False)
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(self.corr_matrix)
        if eigvals.min() < 0:
            self.corr_matrix += (-eigvals.min() + 1e-6) * np.eye(self.latent_dim)
        self._fitted = True
        logger.debug("Gaussian copula fitted on %d samples.", z.shape[0])
        return self

    def sample(self, n: int) -> np.ndarray:
        """Sample from the fitted Gaussian copula.

        Args:
            n: Number of samples.

        Returns:
            Sampled latent vectors, shape (n, latent_dim).
        """
        if not self._fitted:
            raise RuntimeError("Copula must be fitted before sampling. Call fit() first.")
        samples = np.random.multivariate_normal(
            mean=np.zeros(self.latent_dim),
            cov=self.corr_matrix,
            size=n,
        )
        return samples * self.marginal_stds + self.marginal_means

    def sample_tensor(self, n: int, device: torch.device = torch.device("cpu")) -> Tensor:
        """Sample and return as a PyTorch tensor.

        Args:
            n: Number of samples.
            device: Target device.

        Returns:
            Sampled latent tensor, shape (n, latent_dim).
        """
        samples = self.sample(n)
        return torch.from_numpy(samples.astype(np.float32)).to(device)


class StudentTCopula:
    """Student-t copula for modeling heavy-tailed cross-feature dependencies.

    Args:
        latent_dim: Dimensionality of the latent space.
        df: Degrees of freedom for the Student-t distribution.
    """

    def __init__(self, latent_dim: int, df: float = 4.0) -> None:
        self.latent_dim = latent_dim
        self.df = df
        self.corr_matrix: Optional[np.ndarray] = None
        self.marginal_means: Optional[np.ndarray] = None
        self.marginal_stds: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, z: np.ndarray) -> "StudentTCopula":
        """Fit the Student-t copula to latent representations.

        Args:
            z: Latent vectors, shape (N, latent_dim).

        Returns:
            self (for chaining).
        """
        self.marginal_means = z.mean(axis=0)
        self.marginal_stds = z.std(axis=0) + 1e-8
        z_norm = (z - self.marginal_means) / self.marginal_stds
        self.corr_matrix = np.corrcoef(z_norm, rowvar=False)
        eigvals = np.linalg.eigvalsh(self.corr_matrix)
        if eigvals.min() < 0:
            self.corr_matrix += (-eigvals.min() + 1e-6) * np.eye(self.latent_dim)
        self._fitted = True
        return self

    def sample(self, n: int) -> np.ndarray:
        """Sample from the fitted Student-t copula.

        Args:
            n: Number of samples.

        Returns:
            Sampled latent vectors, shape (n, latent_dim).
        """
        if not self._fitted:
            raise RuntimeError("Copula must be fitted before sampling.")

        # Sample from multivariate t-distribution
        # X ~ MVN(0, Σ), W ~ Chi2(df), T = X / sqrt(W/df)
        x = np.random.multivariate_normal(
            mean=np.zeros(self.latent_dim),
            cov=self.corr_matrix,
            size=n,
        )
        w = np.random.chisquare(self.df, size=(n, 1))
        t_samples = x / np.sqrt(w / self.df)

        # Convert to uniform using t CDF, then to Gaussian
        u = stats.t.cdf(t_samples, df=self.df)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z_samples = stats.norm.ppf(u)
        return z_samples * self.marginal_stds + self.marginal_means

    def sample_tensor(self, n: int, device: torch.device = torch.device("cpu")) -> Tensor:
        """Sample and return as a PyTorch tensor."""
        samples = self.sample(n)
        return torch.from_numpy(samples.astype(np.float32)).to(device)


def get_copula(copula_type: str, latent_dim: int, df: float = 4.0):
    """Factory function for creating copula instances.

    Args:
        copula_type: "gaussian" or "student_t".
        latent_dim: Latent space dimensionality.
        df: Degrees of freedom (for Student-t only).

    Returns:
        Copula instance.
    """
    if copula_type == "gaussian":
        return GaussianCopula(latent_dim)
    elif copula_type == "student_t":
        return StudentTCopula(latent_dim, df=df)
    else:
        raise ValueError(f"Unknown copula type: {copula_type!r}. Use 'gaussian' or 'student_t'.")
