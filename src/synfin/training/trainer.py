"""Generic training loop for models with a single ``loss = f(model, batch)`` objective."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from synfin.training.ema import EMA

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
):
    """Build the LR scheduler named by ``cfg["lr_scheduler"]``.

    Supported values: ``None``/``"none"``, ``"cosine"`` (linear warm-up over
    ``warmup_steps`` then cosine decay to 0, stepped per batch) and
    ``"reduce_on_plateau"`` (``lr_scheduler_patience`` / ``lr_scheduler_factor``,
    stepped per epoch on the validation — or training — loss).

    Returns:
        Tuple ``(scheduler, interval)`` with interval "step", "plateau" or None.
    """
    name = (cfg.get("lr_scheduler") or "none").lower()
    if name == "none":
        return None, None
    if name == "cosine":
        total = max(1, cfg.get("epochs", 1) * steps_per_epoch)
        warmup = min(cfg.get("warmup_steps", 0), total - 1)

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), "step"
    if name == "reduce_on_plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg.get("lr_scheduler_patience", 10),
            factor=cfg.get("lr_scheduler_factor", 0.5),
        )
        return sched, "plateau"
    raise ValueError(f"Unknown lr_scheduler {name!r}; use none, cosine or reduce_on_plateau.")


class Trainer:
    """Training loop for DiffusionModel, DiffusionTS and VAECopula.

    Handles gradient clipping, an optional EMA of the weights, an optional LR
    scheduler, deterministic validation, TensorBoard logging and keeping the
    best-validation weights. TimeGAN's multi-phase training uses its own
    methods instead.

    Validation draws the same random numbers every epoch (diffusion losses
    sample random timesteps and noise), so successive validation losses are
    comparable and "best" is not decided by sampling luck. When EMA is enabled
    the EMA weights are validated, kept and returned, since those are the
    weights used for sampling.

    Args:
        model: The generative model to train.
        optimizer: Optimizer.
        device: Compute device.
        log_dir: Directory for TensorBoard logs.
        use_tensorboard: Whether to enable TensorBoard logging.
        callbacks: Callables ``(epoch, metrics) -> None`` run after each epoch.
        grad_clip: Max gradient norm (None/0 disables clipping).
        ema_decay: EMA decay (None/0 disables EMA).
        scheduler: Optional LR scheduler.
        scheduler_interval: "step" (per batch) or "plateau" (per epoch on loss).
        restore_best: After training, load the weights with the lowest
            validation loss (requires a validation loader).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        log_dir: str = "logs",
        use_tensorboard: bool = False,
        callbacks: Optional[List[Callable]] = None,
        grad_clip: Optional[float] = None,
        ema_decay: Optional[float] = None,
        scheduler=None,
        scheduler_interval: Optional[str] = None,
        restore_best: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks or []
        self.grad_clip = grad_clip
        self.ema = EMA(self.model, ema_decay) if ema_decay else None
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.restore_best = restore_best
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch: Optional[int] = None
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
        """Run the training loop.

        Args:
            train_loader: DataLoader for training data.
            epochs: Number of training epochs.
            val_loader: Optional validation DataLoader.
            loss_fn: ``(model, batch) -> loss``; defaults to ``model(batch)``.
                ``self.epoch`` holds the current epoch (e.g. for KL annealing).
            log_interval: Log metrics every N epochs.

        Returns:
            History dictionary with train_loss (and val_loss if validating).
        """
        if loss_fn is None:
            loss_fn = lambda model, batch: model(batch)  # noqa: E731
        if val_loader is not None and len(val_loader) == 0:
            val_loader = None

        history: Dict[str, List[float]] = {"train_loss": []}
        if val_loader:
            history["val_loss"] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                loss = loss_fn(self.model, batch)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                if self.scheduler is not None and self.scheduler_interval == "step":
                    self.scheduler.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            metrics: Dict[str, float] = {
                "loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            if val_loader:
                val_loss = self._validate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                metrics["val_loss"] = val_loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    best_state = copy.deepcopy(self._eval_state_dict())

            if self.scheduler is not None and self.scheduler_interval == "plateau":
                self.scheduler.step(metrics.get("val_loss", train_loss))

            if self.writer:
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, epoch)
            for cb in self.callbacks:
                cb(epoch, metrics)
            if (epoch + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d/%d  " + "  ".join(f"{k}={v:.4g}" for k, v in metrics.items()),
                    epoch + 1,
                    epochs,
                )

        if self.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                "Restored best weights from epoch %d (val_loss=%.4f)",
                self.best_epoch + 1,
                self.best_val_loss,
            )
        elif self.ema is not None:
            self.ema.copy_to(self.model)

        if self.writer:
            self.writer.close()
        return history

    def _eval_state_dict(self) -> Dict[str, torch.Tensor]:
        """Weights used for evaluation: the EMA average if enabled."""
        return self.ema.shadow if self.ema is not None else self.model.state_dict()

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, loss_fn: Callable, seed: int = 0) -> float:
        """Average validation loss with a fixed RNG stream (comparable across epochs)."""
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.ema is not None:
                with self.ema.swapped_in(self.model):
                    return self._val_pass(val_loader, loss_fn)
            return self._val_pass(val_loader, loss_fn)

    def _val_pass(self, val_loader: DataLoader, loss_fn: Callable) -> float:
        self.model.eval()
        total = 0.0
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            total += loss_fn(self.model, batch.to(self.device)).item()
        return total / len(val_loader)
