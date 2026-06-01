"""Diffusion-TS: interpretable diffusion for financial time series.

A SOTA-tier upgrade over the vanilla DDPM in
:mod:`synfin.models.diffusion`. Two key differences:

1. **x0-prediction with seasonal-trend decomposition.** Instead of predicting
   the noise epsilon with a conv U-Net, a Transformer backbone predicts the
   *clean* signal ``x0`` directly as ``trend + seasonality + residual`` (see
   :mod:`synfin.models.diffusion_ts.decomposition`). Predicting x0 is what makes
   the Fourier loss below well-defined.

2. **Fourier (frequency-domain) loss.** The training objective adds a term that
   matches the real FFT of the reconstruction against the real FFT of the
   ground-truth signal. This directly targets the spectral structure behind
   financial *stylized facts* — volatility clustering and the slow decay of
   autocorrelations — which a pure time-domain MSE tends to under-fit.

Reference: Yuan & Qiao, "Diffusion-TS: Interpretable Diffusion for General Time
Series Generation", ICLR 2024 (https://arxiv.org/abs/2403.01742).

The public API (``q_sample``, ``training_loss``, ``forward``) mirrors
:class:`~synfin.models.diffusion.diffusion.DiffusionModel` so the model drops
into the existing :class:`~synfin.training.trainer.Trainer` and CLI scripts.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from synfin.models.diffusion.noise_schedule import (
    compute_schedule_constants,
    cosine_beta_schedule,
    linear_beta_schedule,
)
from synfin.models.diffusion_ts.transformer_backbone import TransformerDenoiser

logger = logging.getLogger(__name__)


class DiffusionTS(nn.Module):
    """Interpretable, transformer-based diffusion model for OHLCV sequences.

    Args:
        in_channels: Number of time-series features.
        seq_length: Sequence length (window size).
        num_timesteps: Total diffusion steps T.
        noise_schedule: "linear" or "cosine".
        beta_start: Beta start for the linear schedule.
        beta_end: Beta end for the linear schedule.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder/decoder layers.
        dim_feedforward: Feed-forward width inside each block.
        dropout: Dropout rate.
        time_embed_dim: Sinusoidal timestep-embedding dimension.
        trend_degree: Polynomial degree for the trend head.
        num_harmonics: Number of Fourier harmonics for the seasonality head.
        fourier_loss_weight: Weight lambda on the frequency-domain loss term.
    """

    def __init__(
        self,
        in_channels: int = 8,
        seq_length: int = 30,
        num_timesteps: int = 1000,
        noise_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        trend_degree: int = 3,
        num_harmonics: int = 6,
        fourier_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_timesteps = num_timesteps
        self.fourier_loss_weight = fourier_loss_weight

        if noise_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif noise_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule!r}")

        for k, v in compute_schedule_constants(betas).items():
            self.register_buffer(k, v)

        self.denoiser = TransformerDenoiser(
            in_channels=in_channels,
            seq_length=seq_length,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
            trend_degree=trend_degree,
            num_harmonics=num_harmonics,
        )

    # ------------------------------------------------------------------
    # Forward (noising) process
    # ------------------------------------------------------------------

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Forward diffusion: sample x_t ~ q(x_t | x_0).

        Args:
            x0: Clean input, shape (batch, seq_len, features).
            t: Timestep indices, shape (batch,).
            noise: Optional pre-sampled noise tensor.

        Returns:
            Noisy tensor x_t, shape (batch, seq_len, features).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None]  # type: ignore[index]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]  # type: ignore[index]
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def predict_x0(self, xt: Tensor, t: Tensor) -> Tensor:
        """Predict the clean signal x0 from a noisy input at timestep t."""
        return self.denoiser(xt, t)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    @staticmethod
    def _fourier_loss(x0_pred: Tensor, x0: Tensor) -> Tensor:
        """MSE between the real-FFT spectra of prediction and target.

        Operates along the time axis and compares both real and imaginary
        parts, so it penalizes amplitude *and* phase mismatch across
        frequencies.
        """
        # rfft over the sequence (time) axis -> (batch, freq, features), complex.
        fx = torch.fft.rfft(x0_pred, dim=1)
        fy = torch.fft.rfft(x0, dim=1)
        return nn.functional.mse_loss(torch.view_as_real(fx), torch.view_as_real(fy))

    def training_loss(self, x0: Tensor) -> Tensor:
        """Combined time-domain + Fourier-domain reconstruction loss.

        Args:
            x0: Clean sequences, shape (batch, seq_len, features).

        Returns:
            Scalar loss.
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        x0_pred = self.predict_x0(xt, t)

        time_loss = nn.functional.mse_loss(x0_pred, x0)
        freq_loss = self._fourier_loss(x0_pred, x0)
        return time_loss + self.fourier_loss_weight * freq_loss

    def forward(self, x0: Tensor) -> Tensor:
        """Compute training loss (alias for ``training_loss``)."""
        return self.training_loss(x0)
