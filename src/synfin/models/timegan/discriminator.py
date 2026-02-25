"""Discriminator network: classifies real vs synthetic latent sequences."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):
    """GRU/LSTM-based discriminator network.

    Classifies whether a latent sequence H is real or synthetic.

    Args:
        hidden_dim: Dimension of latent sequences (input size).
        num_layers: Number of RNN layers.
        rnn_type: "gru" or "lstm".
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h: Latent sequences, shape (batch, seq_len, hidden_dim).

        Returns:
            Classification logits, shape (batch, seq_len, 1).
        """
        out, _ = self.rnn(h)
        return self.proj(out)
