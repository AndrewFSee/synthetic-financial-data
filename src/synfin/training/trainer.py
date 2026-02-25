"""Unified Trainer class supporting all three model types."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Unified trainer for TimeGAN, DiffusionModel, and VAECopula.

    Handles:
      - Training loop with epoch/step logging
      - Validation loss tracking
      - Checkpointing (best + periodic)
      - TensorBoard logging

    Args:
        model: The generative model to train.
        optimizer: Primary optimizer.
        device: Compute device.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for TensorBoard logs.
        use_tensorboard: Whether to enable TensorBoard logging.
        callbacks: List of callback callables (epoch, metrics) -> None.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_tensorboard: bool = False,
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = callbacks or []
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                logger.warning("TensorBoard not available. Skipping TB logging.")

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Run the generic training loop.

        This method is suitable for models with a simple forward → loss interface
        (e.g., DiffusionModel, VAECopula). For TimeGAN's multi-phase training,
        use the model's own training methods directly.

        Args:
            train_loader: DataLoader for training data.
            epochs: Number of training epochs.
            val_loader: Optional validation DataLoader.
            loss_fn: Loss function (model, batch) -> Tensor. Defaults to model.forward.
            log_interval: Log metrics every N epochs.

        Returns:
            History dictionary with train_loss (and optionally val_loss).
        """
        if loss_fn is None:
            loss_fn = lambda model, batch: model(batch)  # noqa: E731

        history: Dict[str, List[float]] = {"train_loss": []}
        if val_loader:
            history["val_loss"] = []

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                loss = loss_fn(self.model, batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            metrics: Dict[str, float] = {"loss": train_loss}

            # --- Validation ---
            if val_loader:
                val_loss = self._validate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                metrics["val_loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best", epoch, metrics)

            # --- TensorBoard ---
            if self.writer:
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, epoch)

            # --- Callbacks ---
            for cb in self.callbacks:
                cb(epoch, metrics)

            if (epoch + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d/%d  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                    epoch + 1, epochs,
                )

        if self.writer:
            self.writer.close()

        return history

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            loss = loss_fn(self.model, batch)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    def _save_checkpoint(
        self,
        tag: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        """Save a model checkpoint."""
        path = self.checkpoint_dir / f"model_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.debug("Saved checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint and restore model/optimizer state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Checkpoint dictionary with epoch and metrics.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
        return ckpt
