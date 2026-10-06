#!/usr/bin/env python
"""End-to-end smoke test for the full synfin pipeline (no network required).

Trains every generative model for a handful of steps on small *dummy* OHLCV-shaped
data, generates samples, and runs the full evaluation suite on each. This is a
fast, deterministic sanity check that the train -> generate -> evaluate path is
wired up correctly for all four models. It verifies *plumbing*, not data quality.

Run:  python scripts/smoke_test.py
Exits non-zero if any model fails.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from synfin.data.dataset import OHLCVDataset
from synfin.data.preprocess import RETURN_FEATURES
from synfin.evaluation.metrics import compute_all_metrics
from synfin.utils.seed import seed_everything

# Small, fast configuration.
N_WINDOWS = 80  # > 50 so the TSTR benchmark also exercises
SEQ_LEN = 16
N_FEATURES = len(RETURN_FEATURES)
EPOCHS = 2

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def make_dummy_data() -> np.ndarray:
    """Create windows shaped like the preprocessed ``returns`` feature set.

    Channels follow RETURN_FEATURES (LogReturn, OpenGap, HighRange, LowRange,
    LogVolumeRel), already standardized as the real pipeline does, with signed
    returns so the downstream TSTR classifier sees both classes.

    Returns:
        Array of shape (N_WINDOWS, SEQ_LEN, N_FEATURES).
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_WINDOWS, SEQ_LEN, N_FEATURES))
    data[:, :, 2:4] = np.abs(data[:, :, 2:4])  # range features are non-negative
    return data.astype(np.float32)


def _loader(data: np.ndarray) -> DataLoader:
    return DataLoader(OHLCVDataset(data), batch_size=16, shuffle=True)


def train_timegan(data, device):
    from synfin.models.timegan import TimeGAN

    model = TimeGAN(input_dim=N_FEATURES, hidden_dim=8, num_layers=2).to(device)
    loader = _loader(data)
    ae_opt = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=1e-3
    )
    model.train_autoencoder(loader, ae_opt, epochs=EPOCHS, device=device)
    sv_opt = torch.optim.Adam(model.supervisor.parameters(), lr=1e-3)
    model.train_supervisor_phase(loader, sv_opt, epochs=EPOCHS, device=device)
    g_opt = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=1e-3
    )
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    e_opt = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=1e-3
    )
    model.train_joint(loader, g_opt, d_opt, e_opt, epochs=EPOCHS, device=device)
    return model.generate(N_WINDOWS, SEQ_LEN, device=device).cpu().numpy()


def train_diffusion(data, device):
    from synfin.models.diffusion import DiffusionModel
    from synfin.models.diffusion.sampler import sample
    from synfin.training.trainer import Trainer

    model = DiffusionModel(
        in_channels=N_FEATURES,
        seq_length=SEQ_LEN,
        num_timesteps=20,
        noise_schedule="linear",
        hidden_dims=[16, 32, 16],
        time_embed_dim=16,
        num_res_blocks=1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    hist = Trainer(model, opt, device=device).train(_loader(data), epochs=EPOCHS)
    _report_loss("diffusion", hist)
    return (
        sample(model, N_WINDOWS, SEQ_LEN, method="ddim", ddim_steps=5, device=device).cpu().numpy()
    )


def train_diffusion_ts(data, device):
    from synfin.models.diffusion_ts import DiffusionTS
    from synfin.models.diffusion_ts.sampler import sample
    from synfin.training.trainer import Trainer

    model = DiffusionTS(
        in_channels=N_FEATURES,
        seq_length=SEQ_LEN,
        num_timesteps=20,
        noise_schedule="linear",
        d_model=16,
        n_heads=2,
        n_layers=1,
        dim_feedforward=32,
        time_embed_dim=16,
        trend_degree=2,
        num_harmonics=3,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    hist = Trainer(model, opt, device=device).train(_loader(data), epochs=EPOCHS)
    _report_loss("diffusion_ts", hist)
    return (
        sample(model, N_WINDOWS, SEQ_LEN, method="ddim", ddim_steps=5, device=device).cpu().numpy()
    )


def train_vae_copula(data, device):
    from synfin.models.vae_copula import VAECopula

    model = VAECopula(
        input_dim=N_FEATURES,
        hidden_dim=16,
        latent_dim=8,
        seq_length=SEQ_LEN,
        num_layers=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.training_step(_loader(data), opt, epochs=EPOCHS, kl_annealing=False, device=device)
    model.fit_latent_sampler(_loader(data), device=device)
    return model.generate(N_WINDOWS, device=device).cpu().numpy()


def _report_loss(name: str, hist: dict) -> None:
    losses = hist.get("train_loss", [])
    if len(losses) >= 2:
        print(f"    {name}: train_loss {losses[0]:.4f} -> {losses[-1]:.4f}")


MODELS = {
    "timegan": train_timegan,
    "diffusion": train_diffusion,
    "diffusion_ts": train_diffusion_ts,
    "vae_copula": train_vae_copula,
}


def main() -> None:
    seed_everything(42)
    device = torch.device("cpu")
    real = make_dummy_data()
    print(f"Dummy data: {real.shape}  (N, seq_len, features)\n")

    failures = []
    for name, trainer in MODELS.items():
        print(f"[{name}] training -> generating -> evaluating ...")
        try:
            synthetic = trainer(real, device)
            assert synthetic.shape == real.shape, f"shape {synthetic.shape} != {real.shape}"
            report = compute_all_metrics(
                real, synthetic, feature_names=RETURN_FEATURES, run_tstr=True
            )
            print(
                f"    OK  realism_score={report['realism_score']:.3f}  "
                f"mmd={report['mmd']:.4f}\n"
            )
        except Exception as exc:  # noqa: BLE001 - smoke test surfaces any failure
            failures.append(name)
            print(f"    FAILED: {type(exc).__name__}: {exc}\n")

    if failures:
        print(f"SMOKE TEST FAILED for: {', '.join(failures)}")
        sys.exit(1)
    print("SMOKE TEST PASSED for all models.")


if __name__ == "__main__":
    main()
