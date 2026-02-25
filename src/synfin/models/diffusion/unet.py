"""1D U-Net denoiser backbone for time-series diffusion."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep t.

    Args:
        embed_dim: Embedding dimension.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timestep indices.

        Args:
            t: Timestep tensor of shape (batch,) with integer values in [0, T).

        Returns:
            Embedding tensor of shape (batch, embed_dim).
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / (half_dim - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class ResidualBlock1D(nn.Module):
    """1D residual block with group normalization and time embedding.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_embed_dim: Dimension of time embedding.
        groups: Number of groups for GroupNorm.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, in_channels, seq_len).
            t_emb: Time embedding, shape (batch, time_embed_dim).

        Returns:
            Output tensor, shape (batch, out_channels, seq_len).
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNet1D(nn.Module):
    """1D U-Net for multivariate time-series denoising.

    The input is (batch, seq_len, in_channels); we transpose to
    (batch, in_channels, seq_len) for Conv1d layers, then transpose back.

    Args:
        in_channels: Number of input features.
        hidden_dims: Channel sizes at each encoder/decoder level.
        time_embed_dim: Dimensionality of the time embedding.
        num_res_blocks: Number of residual blocks at each resolution.
        dropout: Dropout rate.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dims: Optional[List[int]] = None,
        time_embed_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 128, 64]

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_dims[0], 1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        ch = hidden_dims[0]
        encoder_channels = [ch]
        mid_idx = len(hidden_dims) // 2

        for i, out_ch in enumerate(hidden_dims[: mid_idx + 1]):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock1D(ch, out_ch, time_embed_dim, groups, dropout)
                )
                ch = out_ch
            if i < mid_idx:
                encoder_channels.append(ch)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i, out_ch in enumerate(hidden_dims[mid_idx + 1 :]):
            skip_ch = encoder_channels.pop()
            for j in range(num_res_blocks):
                in_ch = ch + skip_ch if j == 0 else out_ch
                self.decoder_blocks.append(
                    ResidualBlock1D(in_ch, out_ch, time_embed_dim, groups, dropout)
                )
                ch = out_ch

        # Output projection
        self.output_norm = nn.GroupNorm(min(groups, ch), ch)
        self.output_proj = nn.Conv1d(ch, in_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict noise given noisy input and timestep.

        Args:
            x: Noisy input, shape (batch, seq_len, in_channels).
            t: Timestep indices, shape (batch,).

        Returns:
            Predicted noise, shape (batch, seq_len, in_channels).
        """
        # (batch, seq_len, C) -> (batch, C, seq_len)
        x = x.transpose(1, 2)
        t_emb = self.time_embed(t)

        h = self.input_proj(x)
        skips = [h]

        mid_idx = len(self.encoder_blocks) // 2 + 1
        block_idx = 0

        for block in self.encoder_blocks:
            h = block(h, t_emb)
            if block_idx < mid_idx - 1:
                skips.append(h)
            block_idx += 1

        num_res_blocks = (
            len(self.decoder_blocks) // max(len(skips), 1)
        ) or 1
        for i, block in enumerate(self.decoder_blocks):
            if i % num_res_blocks == 0 and skips:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        h = self.act(self.output_norm(h))
        out = self.output_proj(h)
        # (batch, C, seq_len) -> (batch, seq_len, C)
        return out.transpose(1, 2)
