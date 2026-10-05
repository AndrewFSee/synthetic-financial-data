"""VAE Decoder: latent z → reconstructed time-series sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Decoder(nn.Module):
    """Recurrent VAE decoder.

    Maps latent vector z → reconstructed sequences X̂.

    The decoder repeats z at each timestep as RNN input to generate
    a sequence of the desired length.

    With ``heteroscedastic=True`` it also predicts a per-step, per-feature
    log-variance, i.e. the parameters of a Gaussian observation model
    N(mean_t, exp(log_var_t)). Sampling from that model is what gives
    generated returns their step-to-step noise; the mean alone is a smooth,
    strongly autocorrelated curve. Because log_var depends on z, it can also
    carry the window's volatility regime.

    With ``ar_noise=True`` the noise is AR(1) instead of independent per step::

        x_t = mean_t + sigma_t * u_t,   u_t = rho * u_{t-1} + sqrt(1 - rho^2) * eps_t

    with a learned persistence ``rho`` per feature (``u`` keeps unit
    variance, so ``sigma`` still sets the scale). Features whose deviations
    persist, such as relative volume, learn rho > 0; near-white-noise returns
    learn rho ~ 0.

    Args:
        latent_dim: Dimensionality of the latent space.
        hidden_dim: RNN hidden dimension.
        output_dim: Number of output features per timestep.
        seq_length: Length of the output sequence.
        num_layers: Number of RNN layers.
        rnn_type: "lstm" or "gru".
        dropout: Dropout rate.
        heteroscedastic: Also predict a per-step log-variance.
        ar_noise: Use AR(1) observation noise (requires heteroscedastic).
    """

    LOG_VAR_RANGE = (-10.0, 4.0)
    MAX_ABS_RHO = 0.99

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 8,
        seq_length: int = 30,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        heteroscedastic: bool = False,
        ar_noise: bool = False,
    ) -> None:
        super().__init__()
        if ar_noise and not heteroscedastic:
            raise ValueError("ar_noise requires heteroscedastic=True.")
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.heteroscedastic = heteroscedastic

        # Project z to hidden state initialization
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)

        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.log_var_proj = nn.Linear(hidden_dim, output_dim) if heteroscedastic else None
        # Unconstrained persistence; rho = MAX_ABS_RHO * tanh(raw), starting at 0.
        self.noise_rho_raw = nn.Parameter(torch.zeros(output_dim)) if ar_noise else None

    def noise_rho(self) -> Tensor:
        """Per-feature AR(1) persistence of the observation noise (zeros if disabled)."""
        if self.noise_rho_raw is None:
            return torch.zeros(self.output_proj.out_features, device=self.output_proj.weight.device)
        return self.MAX_ABS_RHO * torch.tanh(self.noise_rho_raw)

    def nll(self, x: Tensor, z: Tensor) -> Tensor:
        """Gaussian negative log-likelihood of ``x`` given ``z``, averaged per element.

        With AR(1) noise this is the exact likelihood of the AR process: the
        first step is N(mean_0, sigma_0^2); later steps are conditioned on the
        previous standardized residual. The constant 0.5 * log(2 * pi) is omitted.

        Args:
            x: Sequences, shape (batch, seq_length, output_dim).
            z: Latent vectors, shape (batch, latent_dim).

        Returns:
            Scalar mean NLL per data element.
        """
        mean, log_var = self.forward_dist(z)
        e = (x - mean) * torch.exp(-0.5 * log_var)  # standardized residuals
        rho = self.noise_rho()
        first = 0.5 * (log_var[:, :1] + e[:, :1].pow(2))
        scale = 1.0 - rho.pow(2)
        innov = e[:, 1:] - rho * e[:, :-1]
        rest = 0.5 * (log_var[:, 1:] + torch.log(scale) + innov.pow(2) / scale)
        return torch.cat([first, rest], dim=1).mean()

    def sample(self, z: Tensor) -> Tensor:
        """Draw sequences from the observation model given ``z``."""
        mean, log_var = self.forward_dist(z)
        rho = self.noise_rho()
        eps = torch.randn_like(mean)
        u = torch.empty_like(mean)
        u[:, 0] = eps[:, 0]
        innov_scale = torch.sqrt(1.0 - rho.pow(2))
        for t in range(1, mean.shape[1]):
            u[:, t] = rho * u[:, t - 1] + innov_scale * eps[:, t]
        return mean + torch.exp(0.5 * log_var) * u

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vector to the mean sequence.

        Args:
            z: Latent vectors, shape (batch, latent_dim).

        Returns:
            Reconstructed (mean) sequences X̂, shape (batch, seq_length, output_dim).
        """
        return self.output_proj(self._hidden(z))

    def forward_dist(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Decode to the Gaussian observation model's (mean, log_var).

        Args:
            z: Latent vectors, shape (batch, latent_dim).

        Returns:
            Tuple (mean, log_var), each shape (batch, seq_length, output_dim).
        """
        if self.log_var_proj is None:
            raise RuntimeError("forward_dist requires heteroscedastic=True.")
        out = self._hidden(z)
        log_var = torch.clamp(self.log_var_proj(out), *self.LOG_VAR_RANGE)
        return self.output_proj(out), log_var

    def _hidden(self, z: Tensor) -> Tensor:
        """RNN hidden states for every output step."""
        # Repeat z for each timestep
        z_repeated = z.unsqueeze(1).expand(-1, self.seq_length, -1)

        # Initialize hidden state from z
        h0 = torch.tanh(self.fc_hidden(z))
        h0 = h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        if self.rnn_type == "lstm":
            c0 = torch.zeros_like(h0)
            out, _ = self.rnn(z_repeated, (h0, c0))
        else:
            out, _ = self.rnn(z_repeated, h0)
        return out
