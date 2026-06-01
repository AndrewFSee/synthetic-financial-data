"""Tests for the Diffusion-TS model."""

import pytest
import torch

from synfin.models.diffusion_ts import DiffusionTS
from synfin.models.diffusion_ts.decomposition import DecompositionHead
from synfin.models.diffusion_ts.sampler import sample
from synfin.models.diffusion_ts.transformer_backbone import TransformerDenoiser


@pytest.fixture
def small_diffusion_ts():
    return DiffusionTS(
        in_channels=5,
        seq_length=10,
        num_timesteps=20,
        noise_schedule="linear",
        d_model=16,
        n_heads=2,
        n_layers=1,
        dim_feedforward=32,
        time_embed_dim=16,
        trend_degree=2,
        num_harmonics=3,
    )


def test_decomposition_head_shape():
    """DecompositionHead maps hidden states to (batch, seq_len, out_channels)."""
    head = DecompositionHead(d_model=16, out_channels=5, trend_degree=2, num_harmonics=3)
    h = torch.randn(4, 10, 16)
    out = head(h)
    assert out.shape == (4, 10, 5)


def test_transformer_denoiser_predicts_x0_shape():
    """Backbone predicts a clean signal with the same shape as the input."""
    net = TransformerDenoiser(
        in_channels=5, seq_length=10, d_model=16, n_heads=2, n_layers=1, time_embed_dim=16
    )
    x = torch.randn(4, 10, 5)
    t = torch.randint(0, 20, (4,))
    out = net(x, t)
    assert out.shape == x.shape


def test_diffusion_ts_q_sample(small_diffusion_ts):
    """q_sample produces correct shape."""
    x0 = torch.randn(4, 10, 5)
    t = torch.randint(0, 20, (4,))
    xt = small_diffusion_ts.q_sample(x0, t)
    assert xt.shape == x0.shape


def test_diffusion_ts_training_loss(small_diffusion_ts):
    """training_loss returns a finite positive scalar."""
    x0 = torch.randn(4, 10, 5)
    loss = small_diffusion_ts.training_loss(x0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_diffusion_ts_loss_backward(small_diffusion_ts):
    """The combined loss is differentiable end-to-end."""
    x0 = torch.randn(4, 10, 5)
    loss = small_diffusion_ts.training_loss(x0)
    loss.backward()
    grads = [p.grad for p in small_diffusion_ts.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_diffusion_ts_ddim_sample(small_diffusion_ts):
    """DDIM sampler produces correct shape."""
    samples = sample(small_diffusion_ts, num_samples=4, seq_length=10, method="ddim", ddim_steps=5)
    assert samples.shape == (4, 10, 5)


def test_diffusion_ts_ddpm_sample(small_diffusion_ts):
    """DDPM sampler produces correct shape."""
    samples = sample(small_diffusion_ts, num_samples=2, seq_length=10, method="ddpm")
    assert samples.shape == (2, 10, 5)
