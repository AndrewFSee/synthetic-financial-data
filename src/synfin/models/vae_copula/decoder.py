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

    Args:
        latent_dim: Dimensionality of the latent space.
        hidden_dim: RNN hidden dimension.
        output_dim: Number of output features per timestep.
        seq_length: Length of the output sequence.
        num_layers: Number of RNN layers.
        rnn_type: "lstm" or "gru".
        dropout: Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 8,
        seq_length: int = 30,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

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

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vector to sequence.

        Args:
            z: Latent vectors, shape (batch, latent_dim).

        Returns:
            Reconstructed sequences X̂, shape (batch, seq_length, output_dim).
        """
        batch_size = z.shape[0]

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

        return self.output_proj(out)
