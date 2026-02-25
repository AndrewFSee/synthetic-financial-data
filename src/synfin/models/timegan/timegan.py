"""TimeGAN: full model orchestrator with 3-phase training loop."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from synfin.models.timegan.discriminator import Discriminator
from synfin.models.timegan.embedder import Embedder
from synfin.models.timegan.generator import Generator
from synfin.models.timegan.recovery import Recovery
from synfin.models.timegan.supervisor import Supervisor

logger = logging.getLogger(__name__)


class TimeGAN(nn.Module):
    """TimeGAN: Time-series Generative Adversarial Network.

    Based on: Yoon, J., Jarrett, D., & van der Schaar, M. (NeurIPS 2019).

    Three-phase training:
      1. Autoencoder: train Embedder + Recovery to reconstruct real sequences.
      2. Supervisor: train Supervisor to predict next latent state.
      3. Joint: adversarial training of Generator + Discriminator + Supervisor
         with combined reconstruction, supervised, and adversarial losses.

    Args:
        input_dim: Number of input features.
        hidden_dim: RNN hidden dimension (also the latent/module dimension).
        num_layers: Number of RNN layers.
        noise_dim: Dimension of the generator noise input.
        rnn_type: "gru" or "lstm".
        dropout: RNN dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        num_layers: int = 3,
        noise_dim: Optional[int] = None,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        noise_dim = noise_dim or hidden_dim

        self.embedder = Embedder(input_dim, hidden_dim, num_layers, rnn_type, dropout)
        self.recovery = Recovery(hidden_dim, input_dim, num_layers, rnn_type, dropout)
        self.generator = Generator(noise_dim, hidden_dim, num_layers, rnn_type, dropout)
        self.supervisor = Supervisor(hidden_dim, max(num_layers - 1, 1), rnn_type, dropout)
        self.discriminator = Discriminator(hidden_dim, num_layers, rnn_type, dropout)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def embed(self, x: Tensor) -> Tensor:
        """Embed real sequences into latent space."""
        return self.embedder(x)

    def recover(self, h: Tensor) -> Tensor:
        """Recover real sequences from latent representations."""
        return self.recovery(h)

    def generate_latent(self, z: Tensor) -> Tensor:
        """Generate synthetic latent sequences from noise."""
        e_hat = self.generator(z)
        return self.supervisor(e_hat)

    def discriminate(self, h: Tensor) -> Tensor:
        """Classify latent sequences as real (1) or synthetic (0)."""
        return self.discriminator(h)

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------

    def train_autoencoder(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 200,
        device: torch.device = torch.device("cpu"),
    ) -> list[float]:
        """Phase 1: Autoencoder pretraining (Embedder + Recovery).

        Args:
            dataloader: DataLoader yielding real sequences.
            optimizer: Optimizer for Embedder + Recovery parameters.
            epochs: Number of training epochs.
            device: Compute device.

        Returns:
            List of per-epoch reconstruction losses.
        """
        self.train()
        mse = nn.MSELoss()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x: Tensor = batch.to(device)
                optimizer.zero_grad()
                h = self.embedder(x)
                x_tilde = self.recovery(h)
                loss = 10.0 * torch.sqrt(mse(x, x_tilde))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info("[AE] Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

        return losses

    def train_supervisor_phase(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 200,
        device: torch.device = torch.device("cpu"),
    ) -> list[float]:
        """Phase 2: Supervisor pretraining.

        Args:
            dataloader: DataLoader yielding real sequences.
            optimizer: Optimizer for Supervisor parameters.
            epochs: Number of training epochs.
            device: Compute device.

        Returns:
            List of per-epoch supervised losses.
        """
        self.train()
        mse = nn.MSELoss()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x: Tensor = batch.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    h = self.embedder(x)
                h_hat = self.supervisor(h)
                # Supervisor predicts h[1:] from h[:-1]
                loss = mse(h[:, 1:, :], h_hat[:, :-1, :])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info("[SV] Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

        return losses

    def train_joint(
        self,
        dataloader: DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        optimizer_e: torch.optim.Optimizer,
        epochs: int = 300,
        lambda_e: float = 10.0,
        lambda_s: float = 10.0,
        gamma: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, list[float]]:
        """Phase 3: Joint adversarial training.

        Args:
            dataloader: DataLoader yielding real sequences.
            optimizer_g: Optimizer for Generator + Supervisor.
            optimizer_d: Optimizer for Discriminator.
            optimizer_e: Optimizer for Embedder + Recovery.
            epochs: Number of training epochs.
            lambda_e: Weight for embedding/reconstruction loss.
            lambda_s: Weight for supervised loss.
            gamma: Balance between generator and discriminator losses.
            device: Compute device.

        Returns:
            Dictionary with per-epoch loss histories.
        """
        self.train()
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        histories: dict[str, list[float]] = {"g_loss": [], "d_loss": [], "e_loss": []}

        for epoch in range(epochs):
            g_loss_ep = d_loss_ep = e_loss_ep = 0.0
            for batch in dataloader:
                x: Tensor = batch.to(device)
                batch_size, seq_len, _ = x.shape
                z = torch.randn(batch_size, seq_len, self.noise_dim, device=device)

                # --- Generator step ---
                optimizer_g.zero_grad()
                h = self.embedder(x).detach()
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                h_hat_s = self.supervisor(h)

                # Supervised loss
                g_loss_s = mse(h[:, 1:, :], h_hat_s[:, :-1, :])

                # Adversarial loss
                y_fake = self.discriminator(h_hat)
                g_loss_u = bce(y_fake, torch.ones_like(y_fake))

                # Moments matching
                g_loss_v1 = torch.mean(
                    torch.abs(torch.sqrt(h_hat.var(dim=0) + 1e-6) -
                              torch.sqrt(h.var(dim=0) + 1e-6))
                )
                g_loss_v2 = torch.mean(torch.abs(h_hat.mean(dim=0) - h.mean(dim=0)))
                g_loss = (g_loss_u + gamma * g_loss_u +
                          lambda_s * torch.sqrt(g_loss_s) +
                          g_loss_v1 + g_loss_v2)
                g_loss.backward()
                optimizer_g.step()

                # --- Discriminator step ---
                optimizer_d.zero_grad()
                h = self.embedder(x).detach()
                e_hat = self.generator(z).detach()
                h_hat = self.supervisor(e_hat).detach()

                y_real = self.discriminator(h)
                y_fake = self.discriminator(h_hat)
                d_loss_real = bce(y_real, torch.ones_like(y_real))
                d_loss_fake = bce(y_fake, torch.zeros_like(y_fake))
                d_loss = d_loss_real + d_loss_fake
                if d_loss > 0.15:
                    d_loss.backward()
                    optimizer_d.step()

                # --- Embedder/Recovery step ---
                optimizer_e.zero_grad()
                h = self.embedder(x)
                x_tilde = self.recovery(h)
                h_hat_s = self.supervisor(h)

                e_loss_0 = 10.0 * torch.sqrt(mse(x, x_tilde))
                e_loss_s = mse(h[:, 1:, :], h_hat_s[:, :-1, :])
                e_loss = e_loss_0 + 0.1 * e_loss_s
                e_loss.backward()
                optimizer_e.step()

                g_loss_ep += g_loss.item()
                d_loss_ep += d_loss.item()
                e_loss_ep += e_loss.item()

            n = len(dataloader)
            histories["g_loss"].append(g_loss_ep / n)
            histories["d_loss"].append(d_loss_ep / n)
            histories["e_loss"].append(e_loss_ep / n)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "[Joint] Epoch %d/%d  G=%.4f  D=%.4f  E=%.4f",
                    epoch + 1, epochs,
                    g_loss_ep / n, d_loss_ep / n, e_loss_ep / n,
                )

        return histories

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        seq_length: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Generate synthetic OHLCV sequences.

        Args:
            num_samples: Number of sequences to generate.
            seq_length: Length of each sequence.
            device: Compute device.

        Returns:
            Synthetic sequences, shape (num_samples, seq_length, input_dim).
        """
        self.eval()
        z = torch.randn(num_samples, seq_length, self.noise_dim, device=device)
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        x_hat = self.recovery(h_hat)
        return x_hat
