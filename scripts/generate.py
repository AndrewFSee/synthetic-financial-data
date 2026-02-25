#!/usr/bin/env python
"""Generate synthetic OHLCV data from a trained model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from synfin.utils.device import get_device
from synfin.utils.logging import setup_logging
from synfin.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial data from a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        choices=["timegan", "diffusion", "vae_copula"],
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    seed_everything(args.seed)
    device = get_device(args.device)

    # Load model
    if args.model == "timegan":
        from synfin.models.timegan import TimeGAN
        model = TimeGAN(input_dim=args.input_dim)
    elif args.model == "diffusion":
        from synfin.models.diffusion import DiffusionModel
        model = DiffusionModel(in_channels=args.input_dim, seq_length=args.seq_length)
    elif args.model == "vae_copula":
        from synfin.models.vae_copula import VAECopula
        model = VAECopula(input_dim=args.input_dim, seq_length=args.seq_length)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    logger.info("Loaded model from %s", args.checkpoint)

    # Generate
    logger.info("Generating %d samples of seq_length=%d...", args.num_samples, args.seq_length)
    with torch.no_grad():
        if args.model == "timegan":
            samples = model.generate(args.num_samples, args.seq_length, device)
        elif args.model == "diffusion":
            from synfin.models.diffusion.sampler import sample
            samples = sample(model, args.num_samples, args.seq_length, device=device)
        elif args.model == "vae_copula":
            samples = model.generate(args.num_samples, device)

    samples_np = samples.cpu().numpy()
    logger.info("Generated samples shape: %s", samples_np.shape)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name or f"{args.model}_synthetic"
    output_path = output_dir / f"{name}.npy"
    np.save(output_path, samples_np)
    logger.info("Saved synthetic data to %s", output_path)


if __name__ == "__main__":
    main()
