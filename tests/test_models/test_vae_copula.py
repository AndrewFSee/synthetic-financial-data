"""Tests for VAE+Copula model."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.models.vae_copula import VAECopula
from synfin.models.vae_copula.encoder import Encoder
from synfin.models.vae_copula.decoder import Decoder
from synfin.models.vae_copula.copula import GaussianCopula, StudentTCopula, get_copula


@pytest.fixture
def small_vae():
    return VAECopula(
        input_dim=5,
        hidden_dim=16,
        latent_dim=8,
        seq_length=10,
        num_layers=1,
    )


def test_encoder_shape():
    """Encoder produces mu and log_var of correct shape."""
    enc = Encoder(input_dim=5, hidden_dim=16, latent_dim=8, num_layers=1)
    x = torch.randn(4, 10, 5)
    mu, log_var = enc(x)
    assert mu.shape == (4, 8)
    assert log_var.shape == (4, 8)


def test_decoder_shape():
    """Decoder produces output of correct shape."""
    dec = Decoder(latent_dim=8, hidden_dim=16, output_dim=5, seq_length=10, num_layers=1)
    z = torch.randn(4, 8)
    out = dec(z)
    assert out.shape == (4, 10, 5)


def test_vae_forward(small_vae):
    """VAE forward pass returns (x_recon, mu, log_var) of correct shapes."""
    x = torch.randn(4, 10, 5)
    x_recon, mu, log_var = small_vae(x)
    assert x_recon.shape == x.shape
    assert mu.shape == (4, 8)
    assert log_var.shape == (4, 8)


def test_vae_elbo_loss(small_vae):
    """ELBO loss returns a scalar."""
    x = torch.randn(4, 10, 5)
    x_recon, mu, log_var = small_vae(x)
    loss, recon, kl = small_vae.elbo_loss(x, x_recon, mu, log_var)
    assert loss.ndim == 0
    assert recon.item() >= 0
    assert kl.item() >= 0


def test_vae_generate(small_vae):
    """VAE.generate produces correct shape."""
    samples = small_vae.generate(num_samples=8)
    assert samples.shape == (8, 10, 5)


def test_gaussian_copula():
    """GaussianCopula fit and sample work correctly."""
    z = np.random.randn(100, 8)
    copula = GaussianCopula(latent_dim=8)
    copula.fit(z)
    samples = copula.sample(50)
    assert samples.shape == (50, 8)


def test_student_t_copula():
    """StudentTCopula fit and sample work correctly."""
    z = np.random.randn(100, 8)
    copula = StudentTCopula(latent_dim=8, df=4.0)
    copula.fit(z)
    samples = copula.sample(50)
    assert samples.shape == (50, 8)


def test_copula_factory():
    """get_copula returns correct copula type."""
    g = get_copula("gaussian", 8)
    assert isinstance(g, GaussianCopula)
    t = get_copula("student_t", 8)
    assert isinstance(t, StudentTCopula)
    with pytest.raises(ValueError):
        get_copula("unknown", 8)
