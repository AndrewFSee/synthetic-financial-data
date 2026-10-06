"""VAE Decoder: latent z → reconstructed time-series sequences."""

from __future__ import annotations

import math
from typing import Optional, Tuple

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

    With ``noise="student_t"`` the innovations ``eps_t`` are unit-variance
    Student-t with a learned degrees-of-freedom per feature (GARCH-t style),
    giving fat tails *within* a window. Gaussian innovations only produce
    tails through changes in sigma, which leaves within-window tails too thin
    for daily returns.

    With ``garch_noise=True`` the innovations get GARCH(1,1) volatility
    feedback on top of the decoder's sigma::

        xi_t = sqrt(g_t) * eta_t,   g_t = (1 - alpha - beta) + alpha * xi_{t-1}^2 + beta * g_{t-1}

    with learned alpha, beta per feature and E[g] = 1, so the decoder still
    sets the overall (regime-level) volatility. A large move raises the
    volatility of the following steps, so extremes come in clusters, as in
    real returns, instead of as isolated spikes (which is all fat-tailed
    i.i.d. innovations can produce). The likelihood stays exact because g is
    a deterministic function of past observations.

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
        noise: Innovation distribution, "gaussian" or "student_t" (requires
            heteroscedastic).
        garch_noise: Add GARCH(1,1) volatility feedback to the innovations
            (requires heteroscedastic).
    """

    LOG_VAR_RANGE = (-10.0, 4.0)
    MAX_ABS_RHO = 0.99
    MIN_DF = 2.1  # unit-variance Student-t needs df > 2
    INIT_DF = 6.0  # typical of GARCH-t fits to daily returns
    MAX_PERSISTENCE = 0.99  # alpha + beta
    INIT_ALPHA, INIT_PERSISTENCE = 0.05, 0.85

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
        noise: str = "gaussian",
        garch_noise: bool = False,
    ) -> None:
        super().__init__()
        if garch_noise and not heteroscedastic:
            raise ValueError("garch_noise requires heteroscedastic=True.")
        if ar_noise and not heteroscedastic:
            raise ValueError("ar_noise requires heteroscedastic=True.")
        if noise not in ("gaussian", "student_t"):
            raise ValueError(f"Unknown noise {noise!r}; use 'gaussian' or 'student_t'.")
        if noise == "student_t" and not heteroscedastic:
            raise ValueError("student_t noise requires heteroscedastic=True.")
        self.noise = noise
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
        # Unconstrained tail parameter; df = MIN_DF + exp(raw), starting at INIT_DF. The log
        # scale lets df move by a constant factor per optimizer step.
        init = math.log(self.INIT_DF - self.MIN_DF)
        self.noise_df_raw = (
            nn.Parameter(torch.full((output_dim,), init)) if noise == "student_t" else None
        )
        # GARCH(1,1): persistence = MAX_PERSISTENCE * sigmoid(raw_p), alpha = persistence *
        # sigmoid(raw_a), so 0 < alpha < alpha + beta < MAX_PERSISTENCE always holds.
        if garch_noise:
            raw_p = math.log(self.INIT_PERSISTENCE / (self.MAX_PERSISTENCE - self.INIT_PERSISTENCE))
            frac = self.INIT_ALPHA / self.INIT_PERSISTENCE
            raw_a = math.log(frac / (1 - frac))
            self.garch_raw = nn.Parameter(torch.tensor([[raw_p], [raw_a]]).repeat(1, output_dim))
        else:
            self.garch_raw = None

    def noise_rho(self) -> Tensor:
        """Per-feature AR(1) persistence of the observation noise (zeros if disabled)."""
        if self.noise_rho_raw is None:
            return torch.zeros(self.output_proj.out_features, device=self.output_proj.weight.device)
        return self.MAX_ABS_RHO * torch.tanh(self.noise_rho_raw)

    def noise_df(self) -> Optional[Tensor]:
        """Per-feature Student-t degrees of freedom (None for Gaussian noise)."""
        if self.noise_df_raw is None:
            return None
        return self.MIN_DF + torch.exp(self.noise_df_raw)

    def garch_params(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Per-feature GARCH (alpha, beta), or None if disabled."""
        if self.garch_raw is None:
            return None
        persistence = self.MAX_PERSISTENCE * torch.sigmoid(self.garch_raw[0])
        alpha = persistence * torch.sigmoid(self.garch_raw[1])
        return alpha, persistence - alpha

    def _garch_scales(self, xi: Tensor) -> Tensor:
        """Conditional variance multipliers g_t for innovations ``xi`` (g_0 = 1)."""
        alpha, beta = self.garch_params()
        omega = 1.0 - alpha - beta
        g = [torch.ones_like(xi[:, 0])]
        for t in range(1, xi.shape[1]):
            g.append(omega + alpha * xi[:, t - 1].pow(2) + beta * g[-1])
        return torch.stack(g, dim=1)

    def _innovation_nll(self, eta: Tensor) -> Tensor:
        """-log density of unit-variance innovations, minus 0.5 * log(2 * pi).

        The constant is dropped so the Gaussian case is 0.5 * eta^2 and the
        Student-t case tends to it as df grows.
        """
        df = self.noise_df()
        if df is None:
            return 0.5 * eta.pow(2)
        c2 = (df - 2.0) / df  # eta = c * T with T ~ t(df) has unit variance
        log_density = (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(df * math.pi * c2)
            - (df + 1) / 2 * torch.log1p(eta.pow(2) / (df * c2))
        )
        return -log_density - 0.5 * math.log(2 * math.pi)

    def nll(self, x: Tensor, z: Tensor) -> Tensor:
        """Negative log-likelihood of ``x`` given ``z``, averaged per element.

        Exact likelihood of the observation model: with AR(1) noise, each step
        after the first is conditioned on the previous standardized residual,
        through the innovation eta_t = (e_t - rho e_{t-1}) / sqrt(1 - rho^2).
        The constant 0.5 * log(2 * pi) is omitted.

        Args:
            x: Sequences, shape (batch, seq_length, output_dim).
            z: Latent vectors, shape (batch, latent_dim).

        Returns:
            Scalar mean NLL per data element.
        """
        mean, log_var = self.forward_dist(z)
        e = (x - mean) * torch.exp(-0.5 * log_var)  # standardized residuals
        rho = self.noise_rho()
        scale = 1.0 - rho.pow(2)
        xi = torch.cat([e[:, :1], (e[:, 1:] - rho * e[:, :-1]) / torch.sqrt(scale)], dim=1)
        after_first = (torch.arange(x.shape[1], device=x.device) > 0).to(x.dtype)[:, None]
        ar_term = 0.5 * torch.log(scale) * after_first  # (seq_length, output_dim)
        nll = 0.5 * log_var + ar_term
        if self.garch_raw is not None:
            g = self._garch_scales(xi)
            nll = nll + 0.5 * torch.log(g)
            xi = xi / torch.sqrt(g)
        return (nll + self._innovation_nll(xi)).mean()

    def sample(self, z: Tensor) -> Tensor:
        """Draw sequences from the observation model given ``z``."""
        mean, log_var = self.forward_dist(z)
        rho = self.noise_rho()
        df = self.noise_df()
        if df is None:
            eps = torch.randn_like(mean)
        else:
            t = torch.distributions.StudentT(df).sample(mean.shape[:-1])
            eps = t * torch.sqrt((df - 2.0) / df)
        if self.garch_raw is not None:
            alpha, beta = self.garch_params()
            g = torch.ones_like(eps[:, 0])
            xi = torch.empty_like(eps)
            for t in range(eps.shape[1]):
                if t > 0:
                    g = (1.0 - alpha - beta) + alpha * xi[:, t - 1].pow(2) + beta * g
                xi[:, t] = torch.sqrt(g) * eps[:, t]
            eps = xi
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
