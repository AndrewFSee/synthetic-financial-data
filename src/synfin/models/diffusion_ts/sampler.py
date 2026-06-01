"""Sampling for the x0-predicting Diffusion-TS model.

The vanilla DDPM sampler in :mod:`synfin.models.diffusion.sampler` assumes the
network predicts the noise epsilon. Diffusion-TS predicts the clean signal x0
instead, so the reverse step is expressed via the closed-form posterior mean
``q(x_{t-1} | x_t, x0)``. The public ``sample()`` signature matches the vanilla
sampler so :mod:`scripts.generate` can treat both models uniformly.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from synfin.models.diffusion_ts.diffusion_ts import DiffusionTS

logger = logging.getLogger(__name__)


@torch.no_grad()
def ddpm_sample(
    model: DiffusionTS,
    num_samples: int,
    seq_length: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generate samples using the full DDPM reverse chain (x0-parameterized).

    Args:
        model: Trained DiffusionTS model.
        num_samples: Number of sequences to generate.
        seq_length: Sequence length.
        device: Compute device.

    Returns:
        Generated sequences, shape (num_samples, seq_length, in_channels).
    """
    model.eval()
    x = torch.randn(num_samples, seq_length, model.in_channels, device=device)

    for t_idx in reversed(range(model.num_timesteps)):
        t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
        x0_pred = model.predict_x0(x, t_batch)

        beta_t = model.betas[t_idx]  # type: ignore[index]
        alpha_bar_t = model.alphas_cumprod[t_idx]  # type: ignore[index]
        alpha_bar_prev = model.alphas_cumprod_prev[t_idx]  # type: ignore[index]
        alpha_t = model.alphas[t_idx]  # type: ignore[index]

        # Posterior mean of q(x_{t-1} | x_t, x0).
        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        if t_idx > 0:
            var = model.posterior_variance[t_idx]  # type: ignore[index]
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean

    return x


@torch.no_grad()
def ddim_sample(
    model: DiffusionTS,
    num_samples: int,
    seq_length: int,
    num_steps: int = 50,
    eta: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generate samples using DDIM (accelerated, x0-parameterized).

    Args:
        model: Trained DiffusionTS model.
        num_samples: Number of sequences to generate.
        seq_length: Sequence length.
        num_steps: Number of DDIM sampling steps (< num_timesteps).
        eta: Stochasticity (0 = deterministic, 1 = DDPM-like).
        device: Compute device.

    Returns:
        Generated sequences, shape (num_samples, seq_length, in_channels).
    """
    model.eval()
    T = model.num_timesteps
    step_size = max(T // num_steps, 1)
    timesteps = list(reversed(range(0, T, step_size)))[:num_steps]

    x = torch.randn(num_samples, seq_length, model.in_channels, device=device)

    for i, t_idx in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
        x0_pred = model.predict_x0(x, t_batch)

        alpha_bar_t = model.alphas_cumprod[t_idx]  # type: ignore[index]
        alpha_bar_prev = (
            model.alphas_cumprod[timesteps[i + 1]]  # type: ignore[index]
            if i + 1 < len(timesteps)
            else torch.tensor(1.0, device=device)
        )

        # Recover the implied noise from the x0 prediction, then re-noise.
        eps = (x - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1.0 - alpha_bar_t)
        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )
        direction = torch.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
        x = torch.sqrt(alpha_bar_prev) * x0_pred + direction + noise

    return x


def sample(
    model: DiffusionTS,
    num_samples: int,
    seq_length: int,
    method: str = "ddim",
    ddim_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Unified sampling interface (matches the vanilla diffusion sampler).

    Args:
        model: Trained DiffusionTS model.
        num_samples: Number of sequences to generate.
        seq_length: Sequence length.
        method: "ddpm" or "ddim".
        ddim_steps: Number of DDIM steps (only used when method="ddim").
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
