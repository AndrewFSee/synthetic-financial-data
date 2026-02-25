"""Forward and reverse diffusion processes (DDPM)."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from synfin.models.diffusion.noise_schedule import (
    compute_schedule_constants,
    cosine_beta_schedule,
    linear_beta_schedule,
)
from synfin.models.diffusion.unet import UNet1D

logger = logging.getLogger(__name__)


class DiffusionModel(nn.Module):
    """DDPM-based diffusion model for multivariate financial time series.

    Implements:
      - Forward process: q(x_t | x_0) — add noise
      - Reverse process: p_θ(x_{t-1} | x_t) — denoise with UNet
      - Training: predict noise ε from noisy x_t at timestep t

    Args:
        in_channels: Number of time-series features.
        seq_length: Sequence length (window size).
        num_timesteps: Total diffusion steps T.
        noise_schedule: "linear" or "cosine".
        beta_start: Beta start for linear schedule.
        beta_end: Beta end for linear schedule.
        hidden_dims: UNet channel sizes.
        time_embed_dim: Time embedding dimension.
        num_res_blocks: UNet residual blocks per level.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 8,
        seq_length: int = 30,
        num_timesteps: int = 1000,
        noise_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dims: Optional[list] = None,
        time_embed_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_timesteps = num_timesteps

        # Build noise schedule
        if noise_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif noise_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule!r}")

        constants = compute_schedule_constants(betas)
        for k, v in constants.items():
            self.register_buffer(k, v)

        # Denoiser
        self.denoiser = UNet1D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            time_embed_dim=time_embed_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
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

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def training_loss(self, x0: Tensor) -> Tensor:
        """Compute simplified DDPM training loss (predict noise).

        Args:
            x0: Clean sequences, shape (batch, seq_len, features).

        Returns:
            Scalar MSE loss.
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        predicted_noise = self.denoiser(xt, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    def forward(self, x0: Tensor) -> Tensor:
        """Compute training loss (alias for training_loss)."""
        return self.training_loss(x0)
