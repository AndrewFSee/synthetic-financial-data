"""Noise schedules for diffusion models: linear and cosine beta schedules."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tensor:
    """Linear beta schedule.

    Args:
        num_timesteps: Total number of diffusion steps T.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        Beta tensor of shape (num_timesteps,).
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine beta schedule (Nichol & Dhariwal, 2021).

    Args:
        num_timesteps: Total number of diffusion steps T.
        s: Small offset to prevent beta from being too small near t=0.

    Returns:
        Beta tensor of shape (num_timesteps,).
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps)
    alpha_bar = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=0.0, max=0.999)


def compute_schedule_constants(betas: Tensor) -> Dict[str, Tensor]:
    """Precompute constants derived from the beta schedule.

    Args:
        betas: Beta schedule tensor of shape (T,).

    Returns:
        Dictionary with all precomputed constants needed for forward/reverse diffusion.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "log_one_minus_alphas_cumprod": torch.log(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1),
        "posterior_variance": (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ),
    }
