"""Supervisor network: learns temporal dynamics in latent space."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Supervisor(nn.Module):
    """GRU/LSTM-based supervisor network.

    Predicts the next latent state given the current latent sequence,
    enforcing temporal consistency in the latent space.

    Args:
        hidden_dim: Latent space dimension.
        num_layers: Number of RNN layers (typically num_layers - 1 from main RNNs).
        rnn_type: "gru" or "lstm".
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
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
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h: Latent sequences, shape (batch, seq_len, hidden_dim).

        Returns:
            Supervised latent sequences Ŝ, shape (batch, seq_len, hidden_dim).
        """
        out, _ = self.rnn(h)
        return self.proj(out)
