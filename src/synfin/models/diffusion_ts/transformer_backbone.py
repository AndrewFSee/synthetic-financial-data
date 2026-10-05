"""Transformer encoder-decoder denoiser backbone for Diffusion-TS.

Unlike the convolutional :class:`~synfin.models.diffusion.unet.UNet1D`, this
backbone uses self-attention to capture long-range temporal dependencies, then
emits the clean signal ``x0`` through the interpretable seasonal-trend
:class:`~synfin.models.diffusion_ts.decomposition.DecompositionHead`.

Input/output convention matches the rest of the package:
``(batch, seq_len, features)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from synfin.models.diffusion.unet import SinusoidalTimeEmbedding
from synfin.models.diffusion_ts.decomposition import DecompositionHead


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding over the sequence axis."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)[:, None]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Args:
            x: Input, shape (batch, seq_len, d_model).

        Returns:
            Input with positions added, same shape.
        """
        return x + self.pe[: x.shape[1]][None]  # type: ignore[index]


class TransformerDenoiser(nn.Module):
    """Transformer encoder-decoder that predicts the clean signal ``x0``.

    The noisy sequence is projected to ``d_model`` and processed by a
    Transformer encoder. The timestep embedding is injected as an additive
    conditioning token so the network knows how much noise to remove. A
    Transformer decoder then attends over the encoded memory to produce the
    hidden states consumed by the seasonal-trend decomposition head.

    Args:
        in_channels: Number of time-series features.
        seq_length: Sequence length (used to size positional encoding).
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder and decoder layers.
        dim_feedforward: Feed-forward width inside each block.
        dropout: Dropout rate.
        time_embed_dim: Dimension of the sinusoidal timestep embedding.
        trend_degree: Polynomial degree for the trend head.
        num_harmonics: Number of Fourier harmonics for the seasonality head.
    """

    def __init__(
        self,
        in_channels: int = 8,
        seq_length: int = 30,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        trend_degree: int = 3,
        num_harmonics: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length

        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(seq_length, 64))

        # Timestep embedding mapped into model space, injected additively.
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.head = DecompositionHead(
            d_model,
            in_channels,
            trend_degree=trend_degree,
            num_harmonics=num_harmonics,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the clean signal ``x0`` from a noisy input.

        Args:
            x: Noisy sequence, shape (batch, seq_len, in_channels).
            t: Timestep indices, shape (batch,).

        Returns:
            Predicted clean signal x0, shape (batch, seq_len, in_channels).
        """
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)

        # Inject the diffusion timestep as additive conditioning at every step.
        t_emb = self.time_proj(self.time_embed(t))  # (batch, d_model)
        h = h + t_emb[:, None, :]

        memory = self.encoder(h)
        dec = self.decoder(h, memory)
        return self.head(dec)
