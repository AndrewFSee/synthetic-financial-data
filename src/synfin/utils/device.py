"""Device management: CPU, CUDA, and MPS (Apple Silicon)."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """Get the best available compute device.

    Args:
        device: Device specification:
            - "auto": Automatically pick CUDA > MPS > CPU.
            - "cuda": Use CUDA GPU (raises if unavailable).
            - "mps": Use Apple Silicon MPS (raises if unavailable).
            - "cpu": Use CPU.

    Returns:
        torch.device instance.

    Raises:
        ValueError: If the requested device is unavailable.
    """
    if device == "auto":
        if torch.cuda.is_available():
            selected = torch.device("cuda")
        elif torch.backends.mps.is_available():
            selected = torch.device("mps")
        else:
            selected = torch.device("cpu")
        logger.info("Auto-selected device: %s", selected)
        return selected

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available.")
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    # Allow explicit device strings like "cuda:1"
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device specification: {device!r}") from exc
