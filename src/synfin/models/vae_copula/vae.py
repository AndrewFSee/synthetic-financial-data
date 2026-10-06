"""Full VAE model with reparameterization trick and ELBO loss."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from synfin.models.vae_copula.copula import get_copula
from synfin.models.vae_copula.decoder import Decoder
from synfin.models.vae_copula.encoder import Encoder

logger = logging.getLogger(__name__)


class VAECopula(nn.Module):
    """Variational Autoencoder with optional Copula for financial time series.

    Supports:
      - Standard VAE training (ELBO = reconstruction + KL divergence)
      - Copula-based latent sampling at generation time: after training,
        :meth:`fit_copula` fits a Gaussian or Student-t copula to the encoder's
        posterior means of the training data, and :meth:`generate` samples
        latents from it instead of the N(0, I) prior. This matches the
        aggregate posterior the decoder was actually trained on. The fitted
        copula is stored in registered buffers, so it is saved and restored
        with the ``state_dict``.

    Args:
        input_dim: Number of features per timestep.
        hidden_dim: RNN hidden dimension for encoder/decoder.
        latent_dim: Dimensionality of the latent space.
        seq_length: Sequence length.
        num_layers: Number of RNN layers.
        rnn_type: "lstm" or "gru".
        dropout: Dropout rate.
        kl_weight: Beta parameter (1.0 = standard VAE, >1 = β-VAE).
        recon_loss: Observation model / reconstruction loss:

            * ``"gaussian"`` (default): the decoder predicts a mean and a
              log-variance per step; the loss is the Gaussian negative
              log-likelihood plus KL (a proper ELBO), and :meth:`generate`
              samples ``mean + sigma * eps``. Needed for return-like data,
              where most variance is step-to-step noise that a mean-only
              decoder cannot produce (its samples are smooth, strongly
              autocorrelated and far too calm).
            * ``"mse"`` / ``"mae"``: legacy mean-only decoder; generation
              returns the decoder mean.
        copula_type: "gaussian" or "student_t".
        copula_df: Degrees of freedom of the Student-t copula.
        ar_noise: With the Gaussian observation model, make the noise AR(1)
            with a learned per-feature persistence, so persistent features
            (e.g. relative volume) keep their step-to-step autocorrelation.
        noise: Innovation distribution of the Gaussian observation model's
            noise, "gaussian" or "student_t" (learned df per feature; gives
            fat tails within a window).
        garch_noise: Add GARCH(1,1) volatility feedback to the observation noise,
            so large moves cluster (Gaussian observation model only).
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
        recon_loss: str = "gaussian",
        copula_type: str = "gaussian",
        copula_df: float = 4.0,
        ar_noise: bool = True,
        noise: str = "gaussian",
        garch_noise: bool = False,
    ) -> None:
        super().__init__()
        if recon_loss not in ("gaussian", "mse", "mae"):
            raise ValueError(f"Unknown recon_loss {recon_loss!r}; use 'gaussian', 'mse' or 'mae'.")
        get_copula(copula_type, latent_dim, df=copula_df)  # validates copula_type
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.recon_loss = recon_loss
        self.copula_type = copula_type
        self.copula_df = copula_df
        self.register_buffer("copula_fitted", torch.tensor(False))
        self.register_buffer("copula_corr", torch.eye(latent_dim))
        self.register_buffer("copula_mean", torch.zeros(latent_dim))
        self.register_buffer("copula_std", torch.ones(latent_dim))
        # Encoder posteriors of the training windows, for "posterior" latent sampling.
        # Sized by fit_latent_sampler(); resized on checkpoint load.
        self.register_buffer("posterior_mu", torch.empty(0, latent_dim))
        self.register_buffer("posterior_log_var", torch.empty(0, latent_dim))

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
            heteroscedastic=recon_loss == "gaussian",
            ar_noise=ar_noise and recon_loss == "gaussian",
            noise=noise if recon_loss == "gaussian" else "gaussian",
            garch_noise=garch_noise and recon_loss == "gaussian",
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
        """Legacy loss for ``recon_loss`` "mse"/"mae": reconstruction + β·mean KL.

        For the Gaussian observation model use :meth:`negative_elbo`, which
        needs the decoder's log-variance.

        Args:
            x: Original sequences.
            x_recon: Reconstructed sequences.
            mu: Latent mean.
            log_var: Latent log variance.
            kl_weight: Optional override for beta (KL weight).

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss).
        """
        if self.recon_loss == "gaussian":
            raise ValueError("Use negative_elbo() for the Gaussian observation model.")
        beta = kl_weight if kl_weight is not None else self.kl_weight
        if self.recon_loss == "mae":
            recon_loss = F.l1_loss(x_recon, x)
        else:
            recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def negative_elbo(
        self, x: Tensor, kl_weight: Optional[float] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Training objective for any ``recon_loss``: (loss, recon, kl).

        For "gaussian" this is the negative ELBO per data element:
        Gaussian NLL averaged over all elements, plus β times the latent KL
        (summed over latent dims, averaged over the batch) divided by the
        number of elements per sample, so the reconstruction/KL balance is
        that of a true ELBO.

        Args:
            x: Sequences, shape (batch, seq_len, input_dim).
            kl_weight: Optional override for beta.

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss).
        """
        if self.recon_loss != "gaussian":
            x_recon, mu, log_var = self(x)
            return self.elbo_loss(x, x_recon, mu, log_var, kl_weight)
        beta = kl_weight if kl_weight is not None else self.kl_weight
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder.nll(x, z)
        elements = x.shape[1] * x.shape[2]
        kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)).mean() / elements
        return recon + beta * kl, recon, kl

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
        histories: dict[str, list[float]] = {"loss": [], "recon_loss": [], "kl_loss": []}

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
                loss, recon_loss, kl_loss = self.negative_elbo(x, beta)
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
                    epoch + 1,
                    epochs,
                    epoch_loss / n,
                    epoch_recon / n,
                    epoch_kl / n,
                )

        return histories

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # The stored posteriors are sized by the training set, so adopt the
        # checkpoint's shape before the default (shape-checking) load.
        names = ("posterior_mu", "posterior_log_var")
        for name in names:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # Checkpoints from before posterior sampling existed have no stored
        # posteriors: keep the empty buffers ("auto" then uses the copula).
        for name in names:
            if prefix + name in missing_keys:
                missing_keys.remove(prefix + name)

    @torch.no_grad()
    def fit_latent_sampler(
        self,
        dataloader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Prepare latent sampling from the aggregate posterior of the training data.

        Stores every training window's encoder posterior (for the default
        ``"posterior"`` sampler) and fits the copula (for the ``"copula"``
        sampler). The copula uses one draw ``z = mu + sigma * eps`` per window
        rather than the posterior means alone, which would make sampled
        latents too concentrated.

        Note: the stored posteriors are a lossy encoding of the training
        windows and are saved with the checkpoint.

        Args:
            dataloader: DataLoader with (training) sequences.
            device: Compute device.
        """
        self.eval()
        mus, log_vars = [], []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            mu, log_var = self.encoder(batch.to(device))
            mus.append(mu.cpu())
            log_vars.append(log_var.cpu())
        mu, log_var = torch.cat(mus), torch.cat(log_vars)
        self.posterior_mu = mu.to(self.copula_mean.device)
        self.posterior_log_var = log_var.to(self.copula_mean.device)
        z = (mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)).numpy()
        copula = get_copula(self.copula_type, self.latent_dim, df=self.copula_df).fit(z)
        self.copula_corr.copy_(torch.as_tensor(copula.corr_matrix, dtype=torch.float32))
        self.copula_mean.copy_(torch.as_tensor(copula.marginal_means, dtype=torch.float32))
        self.copula_std.copy_(torch.as_tensor(copula.marginal_stds, dtype=torch.float32))
        self.copula_fitted.fill_(True)
        logger.info("Fitted latent sampler on %d training windows.", len(z))

    def fit_copula(
        self, dataloader: DataLoader, device: torch.device = torch.device("cpu")
    ) -> None:
        """Alias of :meth:`fit_latent_sampler` (kept for backward compatibility)."""
        self.fit_latent_sampler(dataloader, device)

    def _copula(self):
        copula = get_copula(self.copula_type, self.latent_dim, df=self.copula_df)
        copula.corr_matrix = self.copula_corr.cpu().double().numpy()
        copula.marginal_means = self.copula_mean.cpu().double().numpy()
        copula.marginal_stds = self.copula_std.cpu().double().numpy()
        copula._fitted = True
        return copula

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        device: torch.device = torch.device("cpu"),
        temperature: float = 1.0,
        use_copula: bool = True,
        sampler: str = "auto",
    ) -> Tensor:
        """Generate synthetic sequences.

        Latent samplers:

        * ``"posterior"``: pick a training window at random and draw z from its
          encoder posterior N(mu_i, sigma_i^2), i.e. sample the aggregate
          posterior exactly. Reproduces rare regimes (e.g. crash-level
          volatility) at their training frequency; a fitted Gaussian copula
          smooths the non-Gaussian latent distribution and under-samples them.
        * ``"copula"``: the fitted Gaussian / Student-t copula.
        * ``"prior"``: N(0, I).
        * ``"auto"`` (default): posterior if fitted, else copula, else prior.

        Args:
            num_samples: Number of sequences to generate.
            device: Compute device.
            temperature: Scales the latent spread (>1 = more diverse). For the
                posterior sampler it scales each posterior's sigma.
            use_copula: Legacy switch; False forces the prior sampler.
            sampler: "auto", "posterior", "copula" or "prior".

        Returns:
            Synthetic sequences, shape (num_samples, seq_length, input_dim).
        """
        self.eval()
        if sampler not in ("auto", "posterior", "copula", "prior"):
            raise ValueError(f"Unknown sampler {sampler!r}.")
        if not use_copula:
            sampler = "prior"
        has_posterior = len(self.posterior_mu) > 0
        if sampler == "auto":
            if has_posterior:
                sampler = "posterior"
            elif bool(self.copula_fitted):
                sampler = "copula"
            else:
                logger.warning("Latent sampler not fitted; sampling the N(0, I) prior.")
                sampler = "prior"
        if sampler == "posterior" and not has_posterior:
            raise RuntimeError("No stored posteriors; call fit_latent_sampler() first.")
        if sampler == "copula" and not bool(self.copula_fitted):
            raise RuntimeError("Copula not fitted; call fit_latent_sampler() first.")

        if sampler == "posterior":
            idx = torch.randint(0, len(self.posterior_mu), (num_samples,))
            mu = self.posterior_mu[idx].to(device)
            std = torch.exp(0.5 * self.posterior_log_var[idx]).to(device)
            z = mu + temperature * std * torch.randn_like(mu)
        elif sampler == "copula":
            z = self._copula().sample_tensor(num_samples, device=device)
            mean = self.copula_mean.to(device)
            z = mean + (z - mean) * temperature
        else:
            z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
        if self.recon_loss != "gaussian":
            return self.decoder(z)
        return self.decoder.sample(z)
