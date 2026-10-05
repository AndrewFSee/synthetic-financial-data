"""Exponential moving average (EMA) of model weights."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn


class EMA:
    """Keep an exponential moving average of a model's parameters and buffers.

    Uses the standard warm-up ``decay_t = min(decay, (1 + t) / (10 + t))`` so a
    high decay such as 0.9999 still tracks the model during short runs (a fixed
    0.9999 would leave the average at the random initialization for the first
    ~10k steps).

    Args:
        model: Model whose weights are averaged.
        decay: Maximum decay rate.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model's current weights into the average."""
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        """Load the averaged weights into ``model``."""
        model.load_state_dict(self.shadow)

    @contextmanager
    def swapped_in(self, model: nn.Module) -> Iterator[None]:
        """Temporarily use the averaged weights (e.g. for validation)."""
        backup = copy.deepcopy(model.state_dict())
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(backup)
