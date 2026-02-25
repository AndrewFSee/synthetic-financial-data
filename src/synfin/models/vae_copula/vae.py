"""Full VAE model with reparameterization trick and ELBO loss."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from synfin.models.vae_copula.decoder import Decoder
from synfin.models.vae_copula.encoder import Encoder

logger = logging.getLogger(__name__)


class VAECopula(nn.Module):
    """Variational Autoencoder with optional Copula for financial time series.

    Supports:
      - Standard VAE training (ELBO = reconstruction + KL divergence)
      - Copula-based correlated sampling at generation time

    Args:
        input_dim: Number of features per timestep.
        hidden_dim: RNN hidden dimension for encoder/decoder.
        latent_dim: Dimensionality of the latent space.
        seq_length: Sequence length.
        num_layers: Number of RNN layers.
        rnn_type: "lstm" or "gru".
        dropout: Dropout rate.
        kl_weight: Beta parameter (1.0 = standard VAE, >1 = β-VAE).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        seq_length: int = 30,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_length=seq_length,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
        )

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick: z = μ + ε·σ, ε ~ N(0, I).

        Args:
            mu: Mean of the latent distribution, shape (batch, latent_dim).
            log_var: Log variance, shape (batch, latent_dim).

        Returns:
            Sampled latent vector z, shape (batch, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input sequences, shape (batch, seq_len, input_dim).

        Returns:
            Tuple of (x_recon, mu, log_var).
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def elbo_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        log_var: Tensor,
        kl_weight: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the ELBO loss = reconstruction loss + β·KL divergence.

        Args:
            x: Original sequences.
            x_recon: Reconstructed sequences.
            mu: Latent mean.
            log_var: Latent log variance.
            kl_weight: Optional override for beta (KL weight).

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss).
        """
        beta = kl_weight if kl_weight is not None else self.kl_weight
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def training_step(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 300,
        kl_weight: float = 1.0,
        kl_annealing: bool = True,
        kl_annealing_epochs: int = 50,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, list[float]]:
        """Train the VAE.

        Args:
            dataloader: DataLoader with real sequences.
            optimizer: Optimizer.
            epochs: Training epochs.
            kl_weight: Final KL weight (beta).
            kl_annealing: Whether to anneal KL weight from 0 to kl_weight.
            kl_annealing_epochs: Epochs to ramp up KL weight.
            device: Compute device.

        Returns:
            Training history dict.
        """
        histories: dict[str, list[float]] = {
            "loss": [], "recon_loss": [], "kl_loss": []
        }

        for epoch in range(epochs):
            # KL annealing
            if kl_annealing and epoch < kl_annealing_epochs:
                beta = kl_weight * (epoch / kl_annealing_epochs)
            else:
                beta = kl_weight

            epoch_loss = epoch_recon = epoch_kl = 0.0
            self.train()
            for batch in dataloader:
                x: Tensor = batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, log_var = self(x)
                loss, recon_loss, kl_loss = self.elbo_loss(x, x_recon, mu, log_var, beta)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            n = len(dataloader)
            histories["loss"].append(epoch_loss / n)
            histories["recon_loss"].append(epoch_recon / n)
            histories["kl_loss"].append(epoch_kl / n)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "[VAE] Epoch %d/%d  loss=%.4f  recon=%.4f  kl=%.4f",
                    epoch + 1, epochs,
                    epoch_loss / n, epoch_recon / n, epoch_kl / n,
                )

        return histories

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        device: torch.device = torch.device("cpu"),
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate synthetic sequences by sampling from the prior.

        Args:
            num_samples: Number of sequences to generate.
            device: Compute device.
            temperature: Latent space temperature (>1 = more diverse).

        Returns:
            Synthetic sequences, shape (num_samples, seq_length, input_dim).
        """
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
        return self.decoder(z)
