"""Loss functions for generative model training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def adversarial_loss(logits: Tensor, is_real: bool) -> Tensor:
    """Binary cross-entropy adversarial loss (GAN).

    Args:
        logits: Discriminator output logits (any shape).
        is_real: If True, target is 1 (real); if False, target is 0 (fake).

    Returns:
        Scalar loss tensor.
    """
    target = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def reconstruction_loss(
    x_recon: Tensor,
    x_real: Tensor,
    method: str = "mse",
) -> Tensor:
    """Reconstruction loss between predicted and real sequences.

    Args:
        x_recon: Reconstructed tensor.
        x_real: Real target tensor.
        method: "mse" or "mae".

    Returns:
        Scalar loss tensor.
    """
    if method == "mse":
        return F.mse_loss(x_recon, x_real)
    elif method == "mae":
        return F.l1_loss(x_recon, x_real)
    else:
        raise ValueError(f"Unknown reconstruction method: {method!r}. Use 'mse' or 'mae'.")


def kl_divergence_loss(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL divergence between the learned latent distribution and N(0, I).

    KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Args:
        mu: Latent mean, shape (batch, latent_dim).
        log_var: Latent log variance, shape (batch, latent_dim).

    Returns:
        Scalar KL loss (mean over batch and latent dims).
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def supervised_loss(h_real: Tensor, h_supervised: Tensor) -> Tensor:
    """TimeGAN supervisor loss: predict next latent state.

    Args:
        h_real: Real latent sequences, shape (batch, seq_len, hidden_dim).
        h_supervised: Supervisor output, shape (batch, seq_len, hidden_dim).

    Returns:
        Scalar MSE loss over the sequence (excluding last/first step).
    """
    return F.mse_loss(h_real[:, 1:, :], h_supervised[:, :-1, :])


def elbo_loss(
    x_recon: Tensor,
    x_real: Tensor,
    mu: Tensor,
    log_var: Tensor,
    beta: float = 1.0,
    recon_method: str = "mse",
) -> tuple[Tensor, Tensor, Tensor]:
    """Evidence Lower Bound (ELBO) loss for VAE.

    ELBO = -reconstruction_loss - beta * KL_divergence

    Args:
        x_recon: Reconstructed sequences.
        x_real: Real sequences.
        mu: Latent mean.
        log_var: Latent log variance.
        beta: KL weight (beta-VAE parameter).
        recon_method: "mse" or "mae".

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    recon = reconstruction_loss(x_recon, x_real, recon_method)
    kl = kl_divergence_loss(mu, log_var)
    return recon + beta * kl, recon, kl
