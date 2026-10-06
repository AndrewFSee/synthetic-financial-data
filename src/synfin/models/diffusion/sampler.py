"""Sampling / generation logic for trained diffusion models."""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from synfin.models.diffusion.diffusion import DiffusionModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def ddpm_sample(
    model: DiffusionModel,
    num_samples: int,
    seq_length: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generate samples using full DDPM reverse chain.

    Args:
        model: Trained DiffusionModel.
        num_samples: Number of sequences to generate.
        seq_length: Sequence length.
        device: Compute device.

    Returns:
        Generated sequences, shape (num_samples, seq_length, in_channels).
    """
    model.eval()
    in_channels = model.in_channels
    x = torch.randn(num_samples, seq_length, in_channels, device=device)

    for t_idx in reversed(range(model.num_timesteps)):
        t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
        x0_pred, _ = model.predict_x0_eps(x, t_batch)

        # Posterior mean of q(x_{t-1} | x_t, x0): avoids dividing by sqrt(alpha_t),
        # which blows up near t = T where the cosine schedule's betas reach 0.999.
        beta_t = model.betas[t_idx]  # type: ignore[index]
        alpha_bar_t = model.alphas_cumprod[t_idx]  # type: ignore[index]
        alpha_bar_prev = model.alphas_cumprod_prev[t_idx]  # type: ignore[index]
        alpha_t = model.alphas[t_idx]  # type: ignore[index]
        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        x = coef_x0 * x0_pred + coef_xt * x

        if t_idx > 0:
            posterior_var = model.posterior_variance[t_idx]  # type: ignore[index]
            x = x + torch.sqrt(posterior_var) * torch.randn_like(x)

    return x


@torch.no_grad()
def ddim_sample(
    model: DiffusionModel,
    num_samples: int,
    seq_length: int,
    num_steps: int = 50,
    eta: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generate samples using DDIM (accelerated, fewer steps).

    Args:
        model: Trained DiffusionModel.
        num_samples: Number of sequences to generate.
        seq_length: Sequence length.
        num_steps: Number of DDIM sampling steps (< num_timesteps).
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM).
        device: Compute device.

    Returns:
        Generated sequences, shape (num_samples, seq_length, in_channels).
    """
    model.eval()
    in_channels = model.in_channels
    T = model.num_timesteps

    # Select evenly-spaced subset of timesteps
    step_size = T // num_steps
    timesteps = list(reversed(range(0, T, step_size)))[:num_steps]

    x = torch.randn(num_samples, seq_length, in_channels, device=device)

    for i, t_idx in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
        x0_pred, predicted_noise = model.predict_x0_eps(x, t_batch)

        alpha_bar_t = model.alphas_cumprod[t_idx]  # type: ignore[index]
        alpha_bar_prev = (
            model.alphas_cumprod[timesteps[i + 1]]  # type: ignore[index]
            if i + 1 < len(timesteps)
            else torch.tensor(1.0, device=device)
        )

        # DDIM update
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        )
        direction = torch.sqrt(1 - alpha_bar_prev - sigma**2) * predicted_noise
        noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
        x = torch.sqrt(alpha_bar_prev) * x0_pred + direction + noise

    return x


def sample(
    model: DiffusionModel,
    num_samples: int,
    seq_length: int,
    method: str = "ddim",
    ddim_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Unified sampling interface.

    Args:
        model: Trained DiffusionModel.
        num_samples: Number of sequences to generate.
        seq_length: Sequence length.
        method: Sampling method ("ddpm" or "ddim").
        ddim_steps: Number of DDIM steps (only used for method="ddim").
        device: Compute device.

    Returns:
        Generated sequences, shape (num_samples, seq_length, in_channels).
    """
    if method == "ddpm":
        return ddpm_sample(model, num_samples, seq_length, device)
    elif method == "ddim":
        return ddim_sample(model, num_samples, seq_length, num_steps=ddim_steps, device=device)
    else:
        raise ValueError(f"Unknown sampling method: {method!r}. Use 'ddpm' or 'ddim'.")
