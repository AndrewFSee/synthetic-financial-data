"""Train a generative model: ``synfin-train`` / ``python scripts/train.py``.

Outputs:
    <processed_dir>/<TICKER>_windows.npz   scaled train/val/test windows + scaler + feature names
    <checkpoint_dir>/<model>.pt            self-describing checkpoint (synfin.training.checkpoint)

For models trained with a validation split the checkpoint holds the weights
with the lowest validation loss (EMA weights when EMA is enabled); TimeGAN has
no validation objective, so it holds the final weights.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from synfin.data.dataset import OHLCVDataset
from synfin.data.download import load_ohlcv
from synfin.data.preprocess import preprocess
from synfin.models.factory import MODEL_NAMES, create_model, model_kwargs_from_config
from synfin.training.checkpoint import save_checkpoint
from synfin.training.trainer import Trainer, build_scheduler
from synfin.utils.config import load_config
from synfin.utils.device import get_device
from synfin.utils.logging import setup_logging
from synfin.utils.seed import seed_everything

logger = logging.getLogger(__name__)

_INDICATOR_FLAGS = {
    "use_rsi": "rsi",
    "use_macd": "macd",
    "use_bollinger": "bollinger",
    "use_atr": "atr",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a generative model for synthetic financial data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--config", default=None, help="Model config YAML (default: configs/<model>.yaml)"
    )
    parser.add_argument("--ticker", default=None, help="Default: first of data.tickers")
    parser.add_argument("--checkpoint-dir", default=None, help="Default: training.checkpoint_dir")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def prepare_data(cfg: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """Load raw OHLCV, preprocess it and save the window bundle."""
    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    df = load_ohlcv(ticker, interval=data_cfg.get("interval", "1d"), data_dir=raw_dir)
    if df is None:
        raise FileNotFoundError(
            f"No raw data for {ticker} in {raw_dir}. "
            f"Run `synfin-download --tickers {ticker}` first."
        )
    indicators = [name for flag, name in _INDICATOR_FLAGS.items() if data_cfg.get(flag, False)]
    result = preprocess(
        df,
        window_size=data_cfg.get("window_size", 30),
        normalization=data_cfg.get("normalization", "zscore"),
        feature_cols=data_cfg.get("features"),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        feature_set=data_cfg.get("feature_set", "returns"),
        indicators=indicators,
        volume_window=data_cfg.get("volume_window", 20),
    )
    out = Path(data_cfg.get("processed_dir", "data/processed")) / f"{ticker}_windows.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        train=result["train"],
        val=result["val"],
        test=result["test"],
        feature_cols=np.array(result["feature_cols"]),
        feature_set=np.array(result["feature_set"]),
        scaler_center=np.array(result["scaler_params"]["center"]),
        scaler_scale=np.array(result["scaler_params"]["scale"]),
        log_volume_base=np.array(result["log_volume_base"]),
    )
    logger.info("Saved window bundle to %s", out)
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    cfg = load_config(args.config or f"configs/{args.model}.yaml")
    cfg_model = cfg.get("model", {}).get("name")
    if cfg_model and cfg_model != args.model:
        logger.error("Config is for model %r but --model is %r.", cfg_model, args.model)
        sys.exit(1)
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    ticker = args.ticker or (data_cfg.get("tickers") or ["AAPL"])[0]
    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.get("checkpoint_dir", "checkpoints"))

    seed_everything(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    try:
        data = prepare_data(cfg, ticker)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    train_windows, val_windows = data["train"], data["val"]
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        OHLCVDataset(train_windows), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = (
        DataLoader(OHLCVDataset(val_windows), batch_size=batch_size, num_workers=num_workers)
        if len(val_windows)
        else None
    )
    seq_length, input_dim = train_windows.shape[1], train_windows.shape[2]
    logger.info(
        "Data: %d train / %d val windows, seq_len=%d, features=%s",
        len(train_windows),
        len(val_windows),
        seq_length,
        data["feature_cols"],
    )

    model_kwargs = model_kwargs_from_config(args.model, input_dim, seq_length, cfg.get("model", {}))
    model = create_model(args.model, model_kwargs).to(device)
    logger.info("Model: %s (%d parameters)", args.model, sum(p.numel() for p in model.parameters()))

    log_dir = str(Path(train_cfg.get("log_dir", "logs")) / args.model)
    if args.model == "timegan":
        history = _train_timegan(model, train_loader, train_cfg, device)
    elif args.model == "vae_copula":
        history = _train_vae(model, train_loader, val_loader, train_cfg, device, log_dir)
    else:
        history = _train_diffusion(model, train_loader, val_loader, train_cfg, device, log_dir)

    save_checkpoint(
        checkpoint_dir / f"{args.model}.pt",
        model,
        model_name=args.model,
        model_kwargs=model_kwargs,
        config=cfg,
        data={
            "ticker": ticker,
            "feature_cols": list(data["feature_cols"]),
            "feature_set": data["feature_set"],
            "seq_length": int(seq_length),
            "scaler": data["scaler_params"],
            "log_volume_base": data["log_volume_base"],
        },
        history=history,
    )
    logger.info("Training complete.")


def _make_trainer(model, optimizer, train_loader, cfg, device, log_dir) -> Trainer:
    scheduler, interval = build_scheduler(optimizer, cfg, len(train_loader))
    return Trainer(
        model,
        optimizer,
        device=device,
        log_dir=log_dir,
        use_tensorboard=cfg.get("use_tensorboard", False),
        grad_clip=cfg.get("gradient_clip"),
        ema_decay=cfg.get("ema_decay"),
        scheduler=scheduler,
        scheduler_interval=interval,
    )


def _train_diffusion(model, train_loader, val_loader, cfg, device, log_dir):
    """Train DiffusionModel / DiffusionTS with the generic Trainer."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-4))
    trainer = _make_trainer(model, optimizer, train_loader, cfg, device, log_dir)
    return trainer.train(
        train_loader,
        epochs=cfg.get("epochs", 500),
        val_loader=val_loader,
        log_interval=cfg.get("log_interval", 10),
    )


def _train_vae(model, train_loader, val_loader, cfg, device, log_dir):
    """Train VAE+Copula (ELBO with optional KL annealing), then fit the latent copula."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    trainer = _make_trainer(model, optimizer, train_loader, cfg, device, log_dir)
    kl_weight = cfg.get("kl_weight", 1.0)
    anneal = cfg.get("kl_annealing", False)
    anneal_epochs = max(1, cfg.get("kl_annealing_epochs", 50))

    def vae_loss_fn(m, batch):
        beta = kl_weight * min(1.0, trainer.epoch / anneal_epochs) if anneal else kl_weight
        if not m.training:
            beta = kl_weight  # validate on the full objective so epochs are comparable
        return m.negative_elbo(batch, kl_weight=beta)[0]

    history = trainer.train(
        train_loader,
        epochs=cfg.get("epochs", 300),
        val_loader=val_loader,
        loss_fn=vae_loss_fn,
        log_interval=cfg.get("log_interval", 10),
    )
    model.fit_copula(train_loader, device=device)
    return history


def _train_timegan(model, train_loader, cfg, device):
    """Run TimeGAN's three training phases."""
    clip = cfg.get("gradient_clip")
    ae_opt = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=cfg.get("autoencoder_lr", 1e-3),
    )
    logger.info("Phase 1: Autoencoder training (%d epochs)", cfg.get("autoencoder_epochs", 200))
    ae = model.train_autoencoder(
        train_loader, ae_opt, cfg.get("autoencoder_epochs", 200), device, grad_clip=clip
    )

    sv_opt = torch.optim.Adam(model.supervisor.parameters(), lr=cfg.get("supervisor_lr", 1e-3))
    logger.info("Phase 2: Supervisor training (%d epochs)", cfg.get("supervisor_epochs", 200))
    sv = model.train_supervisor_phase(
        train_loader, sv_opt, cfg.get("supervisor_epochs", 200), device, grad_clip=clip
    )

    g_opt = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()),
        lr=cfg.get("generator_lr", 1e-4),
    )
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.get("discriminator_lr", 1e-4))
    e_opt = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=cfg.get("joint_lr", 1e-4),
    )
    logger.info("Phase 3: Joint training (%d epochs)", cfg.get("joint_epochs", 300))
    joint = model.train_joint(
        train_loader,
        g_opt,
        d_opt,
        e_opt,
        cfg.get("joint_epochs", 300),
        lambda_e=cfg.get("lambda_e", 10.0),
        lambda_s=cfg.get("lambda_s", 10.0),
        gamma=cfg.get("gamma", 1.0),
        device=device,
        grad_clip=clip,
    )
    return {"autoencoder_loss": ae, "supervisor_loss": sv, **joint}


if __name__ == "__main__":
    main()
