"""Recovery network: maps latent space back to real space."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Recovery(nn.Module):
    """GRU/LSTM-based recovery network.

    Maps latent sequences H → reconstructed sequences X̂.

    Args:
        hidden_dim: Latent dimension (input to recovery).
        output_dim: Number of output features (same as input_dim of Embedder).
        num_layers: Number of RNN layers.
        rnn_type: "gru" or "lstm".
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
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
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h: Latent sequences, shape (batch, seq_len, hidden_dim).

        Returns:
            Reconstructed sequences X̂, shape (batch, seq_len, output_dim).
        """
        out, _ = self.rnn(h)
        return self.proj(out)
