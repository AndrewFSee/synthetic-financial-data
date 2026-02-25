"""Training callbacks: checkpointing, early stopping, logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: "min" (lower is better) or "max" (higher is better).
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Update the callback with the latest metric value.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info("EarlyStopping triggered after %d epochs.", self.patience)
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset the callback state."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        filename_prefix: Prefix for checkpoint filenames.
        monitor: Metric name to monitor.
        mode: "min" or "max".
        save_best_only: If True, only save when metric improves.
        save_every_n_epochs: Also save periodically (0 = disabled).
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        filename_prefix: str = "model",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_every_n_epochs: int = 0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.best_value: Optional[float] = None
        self.best_path: Optional[Path] = None

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Optional[Path]:
        """Save checkpoint if criteria are met.

        Args:
            model: Model to save.
            epoch: Current epoch number.
            metrics: Dictionary of metric name → value.

        Returns:
            Path where checkpoint was saved, or None if not saved.
        """
        value = metrics.get(self.monitor)
        improved = False

        if value is not None:
            if self.best_value is None:
                improved = True
            elif self.mode == "min" and value < self.best_value:
                improved = True
            elif self.mode == "max" and value > self.best_value:
                improved = True

        save = (not self.save_best_only) or improved
        if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
            save = True

        if save:
            suffix = "best" if improved else f"epoch{epoch + 1}"
            path = self.checkpoint_dir / f"{self.filename_prefix}_{suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                path,
            )
            logger.info("Saved checkpoint to %s (metric=%s)", path, value)
            if improved:
                self.best_value = value
                self.best_path = path
            return path

        return None


class LoggingCallback:
    """Log training metrics to console and optionally to file.

    Args:
        log_interval: Log every N epochs.
        log_file: Optional path for writing logs.
    """

    def __init__(
        self,
        log_interval: int = 1,
        log_file: Optional[str] = None,
    ) -> None:
        self.log_interval = log_interval
        self._file = None
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_file, "a")

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for the current epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric name → value.
        """
        if (epoch + 1) % self.log_interval == 0:
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            msg = f"Epoch {epoch + 1:4d} | {metric_str}"
            logger.info(msg)
            if self._file:
                self._file.write(msg + "\n")
                self._file.flush()

    def close(self) -> None:
        """Close the log file if open."""
        if self._file:
            self._file.close()

    def __del__(self) -> None:
        self.close()


class LearningRateSchedulerCallback:
    """Adjust learning rate based on a metric.

    Wraps torch.optim.lr_scheduler.ReduceLROnPlateau.

    Args:
        optimizer: The optimizer to adjust.
        patience: Epochs to wait before reducing LR.
        factor: Factor by which to reduce LR.
        mode: "min" or "max".
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        mode: str = "min",
        min_lr: float = 1e-6,
    ) -> None:
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
        )

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Step the scheduler with the monitored metric.

        Args:
            epoch: Current epoch (unused but kept for API consistency).
            metrics: Must contain the key the scheduler was initialized with.
        """
        val = metrics.get("val_loss", metrics.get("loss"))
        if val is not None:
            self.scheduler.step(val)
