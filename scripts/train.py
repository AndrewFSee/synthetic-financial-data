#!/usr/bin/env python
"""Train a generative model for synthetic financial data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from synfin.data.dataset import OHLCVDataset
from synfin.utils.config import load_config, merge_configs
from synfin.utils.device import get_device
from synfin.utils.logging import setup_logging
from synfin.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a generative model for synthetic financial data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        choices=["timegan", "diffusion", "vae_copula"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to model config YAML file",
    )
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_data(args, cfg: dict):
    """Load preprocessed windows from disk or create dummy data."""
    import numpy as np
    from synfin.data.preprocess import preprocess
    from synfin.data.download import load_ohlcv

    data_path = Path(args.data_dir) / f"{args.ticker}.parquet"
    if data_path.exists():
        import pandas as pd
        df = pd.read_parquet(data_path)
    else:
        # Try raw data
        from synfin.data.download import load_ohlcv
        df = load_ohlcv(args.ticker, data_dir="data/raw")
        if df is None:
            logging.warning("No data found for %s. Using dummy data.", args.ticker)
            return None, None, None

    data_cfg = cfg.get("data", {})
    result = preprocess(
        df,
        window_size=data_cfg.get("window_size", 30),
        normalization=data_cfg.get("normalization", "minmax"),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
    )
    return result["train"], result["val"], result["feature_cols"]


def build_model(model_name: str, input_dim: int, cfg: dict, seq_length: int):
    """Instantiate the requested model."""
    model_cfg = cfg.get("model", {})

    if model_name == "timegan":
        from synfin.models.timegan import TimeGAN
        return TimeGAN(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 24),
            num_layers=model_cfg.get("num_layers", 3),
            noise_dim=model_cfg.get("noise_dim", model_cfg.get("hidden_dim", 24)),
            rnn_type=model_cfg.get("rnn_type", "gru"),
            dropout=model_cfg.get("dropout", 0.0),
        )
    elif model_name == "diffusion":
        from synfin.models.diffusion import DiffusionModel
        return DiffusionModel(
            in_channels=input_dim,
            seq_length=seq_length,
            num_timesteps=model_cfg.get("num_timesteps", 1000),
            noise_schedule=model_cfg.get("noise_schedule", "cosine"),
        )
    elif model_name == "vae_copula":
        from synfin.models.vae_copula import VAECopula
        return VAECopula(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("encoder_hidden_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 32),
            seq_length=seq_length,
            rnn_type=model_cfg.get("rnn_type", "lstm"),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    default_cfg = {}
    if Path(args.default_config).exists():
        default_cfg = load_config(args.default_config)
    model_cfg = load_config(args.config)
    cfg = merge_configs(default_cfg, model_cfg)

    # Seed
    seed = cfg.get("training", {}).get("seed", 42)
    seed_everything(seed)

    # Device
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    logger.info("Using device: %s", device)

    # Data
    train_windows, val_windows, feature_cols = load_data(args, cfg)

    if train_windows is None:
        logger.error("Could not load data. Exiting.")
        sys.exit(1)

    batch_size = cfg.get("training", {}).get("batch_size", 64)
    train_ds = OHLCVDataset(train_windows)
    val_ds = OHLCVDataset(val_windows) if val_windows is not None else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    seq_length = train_windows.shape[1]
    input_dim = train_windows.shape[2]
    logger.info("Data: %d train windows, seq_len=%d, features=%d",
                len(train_ds), seq_length, input_dim)

    # Model
    model = build_model(args.model, input_dim, cfg, seq_length)
    model = model.to(device)
    logger.info("Model: %s (%d parameters)",
                args.model, sum(p.numel() for p in model.parameters()))

    # Train
    train_cfg = cfg.get("training", {})

    if args.model == "timegan":
        _train_timegan(model, train_loader, train_cfg, device, args.checkpoint_dir)
    elif args.model == "diffusion":
        _train_diffusion(model, train_loader, val_loader, train_cfg, device, args.checkpoint_dir)
    elif args.model == "vae_copula":
        _train_vae(model, train_loader, val_loader, train_cfg, device, args.checkpoint_dir)

    logger.info("Training complete.")


def _train_timegan(model, train_loader, cfg, device, checkpoint_dir):
    """Run TimeGAN three-phase training."""
    logger = logging.getLogger(__name__)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Phase 1: Autoencoder
    ae_opt = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=cfg.get("autoencoder_lr", 1e-3),
    )
    logger.info("Phase 1: Autoencoder training (%d epochs)", cfg.get("autoencoder_epochs", 200))
    model.train_autoencoder(train_loader, ae_opt, cfg.get("autoencoder_epochs", 200), device)

    # Phase 2: Supervisor
    sv_opt = torch.optim.Adam(model.supervisor.parameters(), lr=cfg.get("supervisor_lr", 1e-3))
    logger.info("Phase 2: Supervisor training (%d epochs)", cfg.get("supervisor_epochs", 200))
    model.train_supervisor_phase(train_loader, sv_opt, cfg.get("supervisor_epochs", 200), device)

    # Phase 3: Joint
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
    model.train_joint(train_loader, g_opt, d_opt, e_opt, cfg.get("joint_epochs", 300), device=device)

    torch.save(model.state_dict(), Path(checkpoint_dir) / "timegan_best.pt")
    logger.info("Saved TimeGAN checkpoint.")


def _train_diffusion(model, train_loader, val_loader, cfg, device, checkpoint_dir):
    """Train diffusion model."""
    from synfin.training.trainer import Trainer
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-4))
    trainer = Trainer(model, optimizer, device=device, checkpoint_dir=checkpoint_dir)
    trainer.train(train_loader, epochs=cfg.get("epochs", 500), val_loader=val_loader)
    torch.save(model.state_dict(), Path(checkpoint_dir) / "diffusion_best.pt")


def _train_vae(model, train_loader, val_loader, cfg, device, checkpoint_dir):
    """Train VAE+Copula model."""
    from synfin.training.trainer import Trainer
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    def vae_loss_fn(m, batch):
        x_recon, mu, log_var = m(batch)
        loss, _, _ = m.elbo_loss(batch, x_recon, mu, log_var)
        return loss

    trainer = Trainer(model, optimizer, device=device, checkpoint_dir=checkpoint_dir)
    trainer.train(train_loader, epochs=cfg.get("epochs", 300), val_loader=val_loader,
                  loss_fn=vae_loss_fn)
    torch.save(model.state_dict(), Path(checkpoint_dir) / "vae_copula_best.pt")


if __name__ == "__main__":
    main()
