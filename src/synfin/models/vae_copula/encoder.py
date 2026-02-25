"""VAE Encoder: LSTM/GRU → (mu, log_var) for latent distribution."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Recurrent VAE encoder.

    Maps input sequences X → (μ, log σ²) for the latent distribution q(z|x).

    Args:
        input_dim: Number of input features per timestep.
        hidden_dim: RNN hidden dimension.
        latent_dim: Dimensionality of the latent space.
        num_layers: Number of RNN layers.
        rnn_type: "lstm" or "gru".
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.rnn_type = rnn_type.lower()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input sequences to latent parameters.

        Args:
            x: Input sequences, shape (batch, seq_len, input_dim).

        Returns:
            Tuple of (mu, log_var), each shape (batch, latent_dim).
            Uses the final hidden state of the RNN.
        """
        _, hidden = self.rnn(x)

        # Extract last-layer hidden state
        if self.rnn_type == "lstm":
            h = hidden[0]  # (num_layers, batch, hidden_dim)
        else:
            h = hidden  # (num_layers, batch, hidden_dim)

        h_last = h[-1]  # (batch, hidden_dim)
        return self.fc_mu(h_last), self.fc_log_var(h_last)
