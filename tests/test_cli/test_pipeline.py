"""End-to-end: synfin-train -> synfin-generate -> synfin-evaluate on fake OHLCV data."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from synfin.cli import evaluate, generate, train
from synfin.models.factory import MODEL_NAMES

MODEL_OVERRIDES = {
    "timegan": {
        "model": {"hidden_dim": 8, "num_layers": 1},
        "training": {"autoencoder_epochs": 1, "supervisor_epochs": 1, "joint_epochs": 1},
    },
    "diffusion": {
        "model": {
            "num_timesteps": 20,
            "unet": {"hidden_dims": [8, 16, 8], "time_embed_dim": 8, "num_res_blocks": 1},
        },
        "training": {"epochs": 2, "lr_scheduler": "cosine", "warmup_steps": 2, "ema_decay": 0.99},
        "generation": {"ddim_steps": 5},
    },
    "diffusion_ts": {
        "model": {
            "num_timesteps": 20,
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "dim_feedforward": 16,
            "time_embed_dim": 8,
        },
        "training": {"epochs": 2, "gradient_clip": 1.0},
        "generation": {"ddim_steps": 5},
    },
    "vae_copula": {
        "model": {"hidden_dim": 8, "latent_dim": 4, "num_layers": 1},
        "training": {"epochs": 2, "kl_annealing": True, "kl_annealing_epochs": 2},
    },
}


def _fake_ohlcv(n: int = 420, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    open_ = close * np.exp(rng.standard_normal(n) * 0.003)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * 0.005)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * 0.005)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )


def test_train_rejects_mismatched_config():
    """configs/timegan.yaml declares model.name: timegan, so --model diffusion is refused."""
    with pytest.raises(SystemExit):
        train.main(["--model", "diffusion", "--config", "configs/timegan.yaml"])


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_train_generate_evaluate(model, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fake_ohlcv().to_parquet(raw / "FAKE_1d.parquet")

    cfg = {
        "data": {
            "tickers": ["FAKE"],
            "raw_dir": str(raw),
            "processed_dir": str(tmp_path / "processed"),
            "window_size": 16,
        },
        "training": {
            "seed": 0,
            "device": "cpu",
            "batch_size": 32,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
            "use_tensorboard": False,
        },
        "evaluation": {"output_dir": str(tmp_path / "reports")},
    }
    for section, values in MODEL_OVERRIDES[model].items():
        cfg.setdefault(section, {}).update(values)
    (tmp_path / "default.yaml").write_text(
        yaml.safe_dump(yaml.safe_load(open("configs/default.yaml")))
    )
    cfg["defaults"] = ["default"]
    config_path = tmp_path / f"{model}.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    train.main(["--model", model, "--config", str(config_path)])
    ckpt = tmp_path / "ckpt" / f"{model}.pt"
    bundle = tmp_path / "processed" / "FAKE_windows.npz"
    assert ckpt.exists() and bundle.exists()

    out_dir = tmp_path / "synthetic"
    generate.main(["--checkpoint", str(ckpt), "--num-samples", "60", "--output-dir", str(out_dir)])
    synth_path = out_dir / f"{model}_FAKE.npz"
    with np.load(synth_path) as z:
        windows, names, ohlcv = z["windows"], list(z["feature_names"]), z["ohlcv"]
    assert windows.shape == (60, 16, 5)
    assert names[0] == "LogReturn"
    assert ohlcv.shape == (60, 16, 5)
    assert (ohlcv[..., 2] <= ohlcv[..., 1]).all()  # Low <= High

    report_dir = tmp_path / "reports" / model
    evaluate.main(
        [
            "--real-data",
            str(bundle),
            "--synthetic-data",
            str(synth_path),
            "--config",
            str(config_path),
            "--output",
            str(report_dir),
        ]
    )
    report = json.loads((report_dir / "evaluation_report.json").read_text())
    assert 0.0 <= report["realism_score"] <= 1.0
    assert "stylized_facts_synthetic" in report
    # Barely-trained generators may emit one-class TSTR labels; then TSTR scores 0.
    assert "auc" in report["tstr"]["trtr"]
    assert "tstr" in report["realism_components"]
