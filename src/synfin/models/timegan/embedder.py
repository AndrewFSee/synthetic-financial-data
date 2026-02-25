"""Embedding network: maps real space to latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Embedder(nn.Module):
    """GRU/LSTM-based embedding network.

    Maps real sequences X → latent sequences H.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden dimension of the RNN.
        num_layers: Number of RNN layers.
        rnn_type: "gru" or "lstm".
        dropout: Dropout rate (applied between RNN layers).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Real sequences, shape (batch, seq_len, input_dim).

        Returns:
            Latent sequences H, shape (batch, seq_len, hidden_dim).
        """
        out, _ = self.rnn(x)
        return self.proj(out)
