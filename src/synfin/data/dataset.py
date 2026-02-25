"""PyTorch Dataset classes for OHLCV time-series data."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class OHLCVDataset(Dataset):
    """PyTorch Dataset for sliding-window OHLCV time-series.

    Each item is a tensor of shape (window_size, num_features).

    Args:
        windows: NumPy array of shape (N, window_size, num_features).
        feature_cols: Names of features (for reference).
        device: Optional device to move tensors to.
    """

    def __init__(
        self,
        windows: np.ndarray,
        feature_cols: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.feature_cols = feature_cols or []
        self.device = device

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.windows[idx]
        if self.device is not None:
            sample = sample.to(self.device)
        return sample

    @property
    def num_features(self) -> int:
        """Number of features per timestep."""
        return self.windows.shape[-1]

    @property
    def seq_length(self) -> int:
        """Sequence length (window size)."""
        return self.windows.shape[1]


class MultiTickerDataset(Dataset):
    """Dataset that combines windows from multiple tickers.

    Args:
        datasets: List of OHLCVDataset instances.
    """

    def __init__(self, datasets: List[OHLCVDataset]) -> None:
        if not datasets:
            raise ValueError("At least one dataset must be provided.")

        # Validate that all datasets have compatible shapes
        ref_features = datasets[0].num_features
        ref_seq_len = datasets[0].seq_length
        for i, ds in enumerate(datasets[1:], 1):
            if ds.num_features != ref_features:
                raise ValueError(
                    f"Dataset {i} has {ds.num_features} features, "
                    f"expected {ref_features}."
                )
            if ds.seq_length != ref_seq_len:
                raise ValueError(
                    f"Dataset {i} has seq_length {ds.seq_length}, "
                    f"expected {ref_seq_len}."
                )

        self.datasets = datasets
        self._cumulative_sizes = np.cumsum([len(ds) for ds in datasets])

    def __len__(self) -> int:
        return int(self._cumulative_sizes[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        ds_idx = np.searchsorted(self._cumulative_sizes, idx, side="right")
        if ds_idx > 0:
            idx -= self._cumulative_sizes[ds_idx - 1]
        return self.datasets[ds_idx][idx]

    @property
    def num_features(self) -> int:
        return self.datasets[0].num_features

    @property
    def seq_length(self) -> int:
        return self.datasets[0].seq_length
