"""Generate synthetic windows from a checkpoint: ``synfin-generate``.

The model is rebuilt from the checkpoint's stored constructor kwargs, samples
are mapped back to original feature units with the stored scaler, and the
result is written as ``<output-dir>/<model>_<TICKER>.npz`` with keys
``windows`` (N, seq_len, F), ``feature_names`` and, for the ``returns``
feature set, ``ohlcv`` (N, seq_len, 5) rebuilt bars.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from synfin.data.preprocess import invert_scaling, windows_to_ohlcv
from synfin.training.checkpoint import load_checkpoint
from synfin.utils.device import get_device
from synfin.utils.logging import setup_logging
from synfin.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial data from a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint written by synfin-train")
    parser.add_argument("--num-samples", type=int, default=None, help="Default: config value")
    parser.add_argument("--output-dir", default=None, help="Default: config data.synthetic_dir")
    parser.add_argument("--output-name", default=None, help="Default: <model>_<TICKER>")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="VAE latent temperature")
    parser.add_argument("--no-copula", action="store_true", help="VAE: sample the N(0, I) prior")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


@torch.no_grad()
def sample_model(model, name: str, n: int, seq_length: int, gen_cfg: dict, device) -> torch.Tensor:
    """Draw ``n`` scaled windows from a model."""
    if name == "timegan":
        return model.generate(n, seq_length, device)
    if name == "vae_copula":
        return model.generate(
            n,
            device,
            temperature=gen_cfg.get("temperature", 1.0),
            use_copula=gen_cfg.get("use_copula", True),
        )
    if name == "diffusion":
        from synfin.models.diffusion.sampler import sample
    else:
        from synfin.models.diffusion_ts.sampler import sample
    return sample(
        model,
        n,
        seq_length,
        method=gen_cfg.get("sampler", "ddim"),
        ddim_steps=gen_cfg.get("ddim_steps", 50),
        device=device,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)
    seed_everything(args.seed)
    device = get_device(args.device)

    model, payload = load_checkpoint(args.checkpoint, device=device)
    name, data = payload["model_name"], payload["data"]
    gen_cfg = dict(payload["config"].get("generation", {}))
    overrides = {
        "sampler": args.sampler,
        "ddim_steps": args.ddim_steps,
        "temperature": args.temperature,
    }
    gen_cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.no_copula:
        gen_cfg["use_copula"] = False
    num_samples = args.num_samples or gen_cfg.get("num_samples", 1000)
    seq_length = data["seq_length"]
    logger.info("Loaded %s from %s; generating %d windows", name, args.checkpoint, num_samples)

    chunks = []
    for start in range(0, num_samples, args.batch_size):
        n = min(args.batch_size, num_samples - start)
        chunks.append(sample_model(model, name, n, seq_length, gen_cfg, device).cpu().numpy())
    scaled = np.concatenate(chunks)
    windows = invert_scaling(scaled, data["scaler"]).astype(np.float32)
    feature_names = list(data["feature_cols"])

    out = {"windows": windows, "feature_names": np.array(feature_names)}
    if data.get("feature_set") == "returns":
        for col in ("HighRange", "LowRange"):
            if col in feature_names:
                i = feature_names.index(col)
                windows[..., i] = np.clip(windows[..., i], 0.0, None)
        out["ohlcv"] = windows_to_ohlcv(
            windows, feature_names, log_volume_base=data.get("log_volume_base", 0.0)
        ).astype(np.float32)

    data_cfg = payload["config"].get("data", {})
    output_dir = Path(args.output_dir or data_cfg.get("synthetic_dir", "data/synthetic"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or "{}_{}".format(name, data.get("ticker", "synthetic"))
    path = output_dir / f"{stem}.npz"
    np.savez(path, **out)
    logger.info("Saved %s windows to %s", windows.shape, path)


if __name__ == "__main__":
    main()
