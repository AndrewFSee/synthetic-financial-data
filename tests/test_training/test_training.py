"""Tests for the Trainer, EMA, schedulers, checkpoints, model factory and config loading."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from synfin.data.dataset import OHLCVDataset
from synfin.models.factory import MODEL_NAMES, create_model, model_kwargs_from_config
from synfin.models.vae_copula import VAECopula
from synfin.training.checkpoint import load_checkpoint, save_checkpoint
from synfin.training.ema import EMA
from synfin.training.trainer import Trainer, build_scheduler
from synfin.utils.config import load_config

SEQ, FEAT = 12, 4

SMALL_CFG = {
    "timegan": {"hidden_dim": 8, "num_layers": 1},
    "diffusion": {
        "num_timesteps": 20,
        "unet": {"hidden_dims": [8, 16, 8], "time_embed_dim": 8, "num_res_blocks": 1},
    },
    "diffusion_ts": {
        "num_timesteps": 20,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "dim_feedforward": 16,
        "time_embed_dim": 8,
    },
    "vae_copula": {"hidden_dim": 8, "latent_dim": 4, "num_layers": 1},
}


@pytest.fixture
def loaders():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((64, SEQ, FEAT)).astype(np.float32)
    return (
        DataLoader(OHLCVDataset(data[:48]), batch_size=16, shuffle=True),
        DataLoader(OHLCVDataset(data[48:]), batch_size=16),
    )


def _small(name):
    kwargs = model_kwargs_from_config(name, FEAT, SEQ, SMALL_CFG[name])
    return create_model(name, kwargs), kwargs


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_factory_honours_config(name):
    model, kwargs = _small(name)
    assert isinstance(model, torch.nn.Module)
    if name == "diffusion":
        assert kwargs["hidden_dims"] == [8, 16, 8]
        assert model.denoiser.input_proj.out_channels == 8


def test_factory_unknown_model():
    with pytest.raises(ValueError):
        model_kwargs_from_config("nope", 1, 1, {})


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_checkpoint_roundtrip_rebuilds_identical_model(name, tmp_path):
    """A checkpoint rebuilds the same architecture and weights without the config."""
    torch.manual_seed(0)
    model, kwargs = _small(name)
    path = save_checkpoint(
        tmp_path / "m.pt", model, model_name=name, model_kwargs=kwargs, data={"seq_length": SEQ}
    )
    loaded, payload = load_checkpoint(path)
    assert payload["data"]["seq_length"] == SEQ
    for (k, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k


def test_load_checkpoint_rejects_legacy_format(tmp_path):
    torch.save({"model_state_dict": {}}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="no model metadata"):
        load_checkpoint(tmp_path / "old.pt")


def test_trainer_validation_is_deterministic(loaders):
    """Diffusion validation loss is identical for identical weights (fixed RNG stream)."""
    train_loader, val_loader = loaders
    model, _ = _small("diffusion")
    trainer = Trainer(model, torch.optim.Adam(model.parameters()))
    loss_fn = lambda m, b: m(b)  # noqa: E731
    assert trainer._validate(val_loader, loss_fn) == trainer._validate(val_loader, loss_fn)


def test_trainer_restores_best_weights(loaders):
    """After training, the model holds the weights of the best validation epoch."""
    train_loader, val_loader = loaders
    torch.manual_seed(0)
    model, _ = _small("diffusion_ts")
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-2))
    history = trainer.train(train_loader, epochs=4, val_loader=val_loader)
    best = min(history["val_loss"])
    assert trainer.best_val_loss == pytest.approx(best)
    loss_fn = lambda m, b: m(b)  # noqa: E731
    assert trainer._validate(val_loader, loss_fn) == pytest.approx(best)


def test_trainer_ema_grad_clip_and_scheduler(loaders):
    train_loader, val_loader = loaders
    model, _ = _small("diffusion")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = {"lr_scheduler": "cosine", "warmup_steps": 2, "epochs": 3}
    scheduler, interval = build_scheduler(opt, cfg, len(train_loader))
    trainer = Trainer(
        model, opt, grad_clip=1.0, ema_decay=0.999, scheduler=scheduler, scheduler_interval=interval
    )
    history = trainer.train(train_loader, epochs=3, val_loader=val_loader)
    assert len(history["train_loss"]) == 3
    assert opt.param_groups[0]["lr"] < 1e-3  # cosine decayed
    assert trainer.ema.num_updates == 3 * len(train_loader)


def test_build_scheduler_variants():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    assert build_scheduler(opt, {}, 10) == (None, None)
    assert build_scheduler(opt, {"lr_scheduler": "reduce_on_plateau"}, 10)[1] == "plateau"
    with pytest.raises(ValueError):
        build_scheduler(opt, {"lr_scheduler": "bogus"}, 10)


def test_ema_warmup_tracks_model():
    """With warm-up, even decay=0.9999 follows the model during short runs."""
    lin = torch.nn.Linear(2, 2)
    ema = EMA(lin, decay=0.9999)
    with torch.no_grad():
        lin.weight.add_(1.0)
    for _ in range(50):
        ema.update(lin)
    assert torch.allclose(ema.shadow["weight"], lin.weight, atol=0.05)


def test_vae_copula_fit_and_roundtrip(loaders, tmp_path):
    """The fitted copula is used for sampling and survives a checkpoint round-trip."""
    train_loader, _ = loaders
    torch.manual_seed(0)
    vae = VAECopula(input_dim=FEAT, hidden_dim=8, latent_dim=4, seq_length=SEQ, num_layers=1)
    assert not bool(vae.copula_fitted)
    vae.fit_copula(train_loader)
    assert bool(vae.copula_fitted)
    assert not torch.allclose(vae.copula_corr, torch.eye(4))
    assert vae.generate(10).shape == (10, SEQ, FEAT)

    kwargs = {"input_dim": FEAT, "hidden_dim": 8, "latent_dim": 4, "seq_length": SEQ}
    path = save_checkpoint(
        tmp_path / "vae.pt", vae, model_name="vae_copula", model_kwargs={**kwargs, "num_layers": 1}
    )
    loaded, _ = load_checkpoint(path)
    assert bool(loaded.copula_fitted)
    assert torch.equal(loaded.copula_corr, vae.copula_corr)


def test_vae_student_t_copula_and_mae():
    vae = VAECopula(
        input_dim=FEAT,
        hidden_dim=8,
        latent_dim=4,
        seq_length=SEQ,
        num_layers=1,
        copula_type="student_t",
        recon_loss="mae",
    )
    x = torch.randn(5, SEQ, FEAT)
    recon, mu, log_var = vae(x)
    _, recon_loss, _ = vae.elbo_loss(x, recon, mu, log_var)
    assert torch.isclose(recon_loss, torch.nn.functional.l1_loss(recon, x))
    with pytest.raises(ValueError):
        VAECopula(input_dim=FEAT, recon_loss="huber")


def test_load_config_resolves_defaults(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnested: {x: 1, y: 2}\n")
    (tmp_path / "mid.yaml").write_text("defaults: [base]\nnested: {y: 3}\n")
    (tmp_path / "top.yaml").write_text("defaults:\n  - mid\nb: 2\n")
    cfg = load_config(tmp_path / "top.yaml")
    assert cfg == {"a": 1, "b": 2, "nested": {"x": 1, "y": 3}}


def test_repo_configs_load():
    for name in MODEL_NAMES + ("diffusion_ts_quick",):
        cfg = load_config(f"configs/{name}.yaml")
        assert "defaults" not in cfg
        assert cfg["data"]["feature_set"] == "returns"
