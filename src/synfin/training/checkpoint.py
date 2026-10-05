"""Self-describing model checkpoints.

A checkpoint stores everything needed to rebuild and use the model without the
original config: the model name and constructor kwargs, the training config,
and the data metadata (feature names, sequence length, scaler parameters).
Everything is plain Python / tensors so ``torch.load(weights_only=True)`` works.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    model_name: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Path:
    """Save ``model`` with the metadata needed to rebuild it.

    Args:
        path: Destination file.
        model: Model to save.
        model_name: Factory name (see :mod:`synfin.models.factory`).
        model_kwargs: Constructor kwargs used to build the model.
        config: The merged training config.
        data: Data metadata, e.g. ``feature_cols``, ``seq_length``, ``scaler``.
        **extra: Anything else (epoch, metrics, ...).

    Returns:
        The checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "model_kwargs": model_kwargs or {},
        "config": config or {},
        "data": data or {},
        **extra,
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        device: Device to map the weights to.

    Returns:
        Tuple ``(model, payload)`` with the model in eval mode on ``device``.
    """
    from synfin.models.factory import create_model

    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not payload.get("model_name"):
        raise ValueError(
            f"{path} has no model metadata (it predates self-describing checkpoints). "
            "Retrain the model to produce a loadable checkpoint."
        )
    model = create_model(payload["model_name"], payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload
