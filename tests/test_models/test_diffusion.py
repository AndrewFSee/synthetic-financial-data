"""Tests for Diffusion model."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.models.diffusion import DiffusionModel
from synfin.models.diffusion.noise_schedule import (
    linear_beta_schedule,
    cosine_beta_schedule,
    compute_schedule_constants,
)
from synfin.models.diffusion.sampler import sample


def test_linear_beta_schedule():
    """Linear schedule has correct shape and monotone increase."""
    betas = linear_beta_schedule(100)
    assert betas.shape == (100,)
    assert betas[0] < betas[-1]


def test_cosine_beta_schedule():
    """Cosine schedule has correct shape and values in (0, 1)."""
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert (betas > 0).all()
    assert (betas < 1).all()


def test_schedule_constants():
    """compute_schedule_constants returns all expected keys."""
    betas = linear_beta_schedule(10)
    consts = compute_schedule_constants(betas)
    for key in ["alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"]:
        assert key in consts
        assert consts[key].shape == (10,)


@pytest.fixture
def small_diffusion():
    return DiffusionModel(
        in_channels=5,
        seq_length=10,
        num_timesteps=20,
        noise_schedule="linear",
        hidden_dims=[16, 32, 16],
        time_embed_dim=16,
        num_res_blocks=1,
    )


def test_diffusion_q_sample(small_diffusion):
    """q_sample produces correct shape."""
    x0 = torch.randn(4, 10, 5)
    t = torch.randint(0, 20, (4,))
    xt = small_diffusion.q_sample(x0, t)
    assert xt.shape == x0.shape


def test_diffusion_training_loss(small_diffusion):
    """training_loss returns a scalar."""
    x0 = torch.randn(4, 10, 5)
    loss = small_diffusion.training_loss(x0)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_diffusion_ddim_sample(small_diffusion):
    """DDIM sampler produces correct shape."""
    samples = sample(
        small_diffusion, num_samples=4, seq_length=10, method="ddim", ddim_steps=5
    )
    assert samples.shape == (4, 10, 5)


def test_diffusion_ddpm_sample(small_diffusion):
    """DDPM sampler produces correct shape."""
    samples = sample(
        small_diffusion, num_samples=2, seq_length=10, method="ddpm"
    )
    assert samples.shape == (2, 10, 5)
