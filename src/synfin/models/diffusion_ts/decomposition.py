"""Interpretable seasonal-trend decomposition heads for Diffusion-TS.

The denoiser predicts the *clean* signal ``x0`` as the sum of three
interpretable components (Zhou et al., "Diffusion-TS", ICLR 2024):

    x0_hat = trend + seasonality + residual

  * **trend**     — a smooth, low-order polynomial in time (captures drift).
  * **seasonality** — a sum of learned Fourier (sin/cos) harmonics
                      (captures periodic / cyclical structure).
  * **residual**  — a free-form linear projection of the remaining signal
                      (captures everything the structured parts miss).

Decomposing the output this way makes the generator interpretable *and*
gives the Fourier loss in :mod:`diffusion_ts` a well-conditioned target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TrendHead(nn.Module):
    """Polynomial trend head.

    Maps decoder hidden states to per-feature polynomial coefficients and
    evaluates the polynomial on a normalized time grid ``[0, 1]``.

    Args:
        d_model: Decoder hidden dimension.
        out_channels: Number of output features.
        degree: Polynomial degree (0 = constant, 1 = linear, ...).
    """

    def __init__(self, d_model: int, out_channels: int, degree: int = 3) -> None:
        super().__init__()
        self.degree = degree
        self.out_channels = out_channels
        # One coefficient per (feature, power) predicted from the pooled state.
        self.coef_proj = nn.Linear(d_model, out_channels * (degree + 1))

    def forward(self, h: Tensor) -> Tensor:
        """Evaluate the polynomial trend.

        Args:
            h: Decoder states, shape (batch, seq_len, d_model).

        Returns:
            Trend signal, shape (batch, seq_len, out_channels).
        """
        batch, seq_len, _ = h.shape
        device = h.device

        # Pool over time to obtain global trend coefficients per sequence.
        pooled = h.mean(dim=1)  # (batch, d_model)
        coefs = self.coef_proj(pooled)  # (batch, out_channels * (degree + 1))
        coefs = coefs.view(batch, self.out_channels, self.degree + 1)

        # Normalized time grid in [0, 1]: (seq_len, degree + 1) Vandermonde matrix.
        t = torch.linspace(0.0, 1.0, seq_len, device=device)
        powers = torch.arange(self.degree + 1, device=device)
        vander = t[:, None] ** powers[None, :]  # (seq_len, degree + 1)

        # (batch, out_channels, seq_len) -> (batch, seq_len, out_channels)
        trend = torch.einsum("bcd,sd->bcs", coefs, vander)
        return trend.transpose(1, 2)


class SeasonalityHead(nn.Module):
    """Fourier seasonality head.

    Predicts per-feature amplitudes for a fixed bank of sin/cos harmonics
    and reconstructs the seasonal component on a normalized time grid.

    Args:
        d_model: Decoder hidden dimension.
        out_channels: Number of output features.
        num_harmonics: Number of Fourier harmonics (frequencies).
    """

    def __init__(self, d_model: int, out_channels: int, num_harmonics: int = 6) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics
        self.out_channels = out_channels
        # Two amplitudes (sin, cos) per (feature, harmonic).
        self.amp_proj = nn.Linear(d_model, out_channels * num_harmonics * 2)

    def forward(self, h: Tensor) -> Tensor:
        """Evaluate the Fourier seasonality.

        Args:
            h: Decoder states, shape (batch, seq_len, d_model).

        Returns:
            Seasonal signal, shape (batch, seq_len, out_channels).
        """
        batch, seq_len, _ = h.shape
        device = h.device

        pooled = h.mean(dim=1)  # (batch, d_model)
        amps = self.amp_proj(pooled)
        amps = amps.view(batch, self.out_channels, self.num_harmonics, 2)
        sin_amp, cos_amp = amps[..., 0], amps[..., 1]  # (batch, C, H)

        # Harmonic angular frequencies on a normalized [0, 1] grid.
        t = torch.linspace(0.0, 1.0, seq_len, device=device)
        harmonics = torch.arange(1, self.num_harmonics + 1, device=device).float()
        angles = 2.0 * math.pi * harmonics[None, :] * t[:, None]  # (seq_len, H)
        sin_basis = torch.sin(angles)  # (seq_len, H)
        cos_basis = torch.cos(angles)

        season = torch.einsum("bch,sh->bcs", sin_amp, sin_basis) + torch.einsum(
            "bch,sh->bcs", cos_amp, cos_basis
        )
        return season.transpose(1, 2)


class DecompositionHead(nn.Module):
    """Combine trend, seasonality, and residual into the predicted ``x0``.

    Args:
        d_model: Decoder hidden dimension.
        out_channels: Number of output features.
        trend_degree: Degree of the polynomial trend.
        num_harmonics: Number of Fourier harmonics for seasonality.
    """

    def __init__(
        self,
        d_model: int,
        out_channels: int,
        trend_degree: int = 3,
        num_harmonics: int = 6,
    ) -> None:
        super().__init__()
        self.trend = TrendHead(d_model, out_channels, degree=trend_degree)
        self.seasonality = SeasonalityHead(d_model, out_channels, num_harmonics=num_harmonics)
        # Free-form residual projects each timestep's state directly.
        self.residual = nn.Linear(d_model, out_channels)

    def forward(self, h: Tensor) -> Tensor:
        """Reconstruct the clean signal from decoder states.

        Args:
            h: Decoder states, shape (batch, seq_len, d_model).

        Returns:
            Predicted x0, shape (batch, seq_len, out_channels).
        """
        return self.trend(h) + self.seasonality(h) + self.residual(h)
