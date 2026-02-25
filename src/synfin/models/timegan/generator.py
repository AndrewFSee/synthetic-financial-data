"""Generator network: produces synthetic latent sequences from noise."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    """GRU/LSTM-based generator network.

    Maps random noise Z → synthetic latent sequences Ê.

    Args:
        noise_dim: Dimension of noise input.
        hidden_dim: Hidden dimension.
        num_layers: Number of RNN layers.
        rnn_type: "gru" or "lstm".
        dropout: Dropout rate.
    """

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z: Noise tensor, shape (batch, seq_len, noise_dim).

        Returns:
            Synthetic latent sequences Ê, shape (batch, seq_len, hidden_dim).
        """
        out, _ = self.rnn(z)
        return self.proj(out)
