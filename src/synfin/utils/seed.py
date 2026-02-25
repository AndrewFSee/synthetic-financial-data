"""Reproducibility utilities: seed everything."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch (CPU and CUDA),
    and configures deterministic behavior where possible.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Encourage deterministic algorithms (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.debug("Set random seed to %d", seed)
