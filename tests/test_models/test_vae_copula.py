"""Tests for VAE+Copula model."""

import numpy as np
import pytest
import torch

from synfin.models.vae_copula import VAECopula
from synfin.models.vae_copula.copula import GaussianCopula, StudentTCopula, get_copula
from synfin.models.vae_copula.decoder import Decoder
from synfin.models.vae_copula.encoder import Encoder


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


def test_vae_negative_elbo(small_vae):
    """The Gaussian negative ELBO returns scalars with a non-negative KL."""
    x = torch.randn(4, 10, 5)
    loss, recon, kl = small_vae.negative_elbo(x)
    assert loss.ndim == 0
    assert torch.isfinite(recon)
    assert kl.item() >= 0
    with pytest.raises(ValueError, match="negative_elbo"):
        small_vae.elbo_loss(x, *small_vae(x))


def test_vae_legacy_mse_elbo_loss():
    """The legacy mean-only decoder keeps the old elbo_loss API."""
    vae = VAECopula(
        input_dim=5, hidden_dim=16, latent_dim=8, seq_length=10, num_layers=1, recon_loss="mse"
    )
    vae.eval()  # deterministic latent (z = mu) so both paths see the same z
    x = torch.randn(4, 10, 5)
    loss, recon, kl = vae.elbo_loss(x, *vae(x))
    assert recon.item() >= 0 and kl.item() >= 0
    assert torch.isclose(vae.negative_elbo(x)[0], loss)


def test_ar_noise_nll_matches_exact_gaussian_density():
    """The AR(1) likelihood equals the multivariate normal with cov sigma_i sigma_j rho^|i-j|."""
    from scipy.stats import multivariate_normal

    torch.manual_seed(0)
    dec = Decoder(
        latent_dim=3,
        hidden_dim=8,
        output_dim=1,
        seq_length=6,
        num_layers=1,
        heteroscedastic=True,
        ar_noise=True,
    )
    with torch.no_grad():
        dec.noise_rho_raw.fill_(0.8)
    z = torch.randn(1, 3)
    x = torch.randn(1, 6, 1)
    with torch.no_grad():
        mean, log_var = dec.forward_dist(z)
        nll = dec.nll(x, z).item() * 6  # per-element mean -> total
        rho = dec.noise_rho().item()
    sd = np.exp(0.5 * log_var[0, :, 0].numpy())
    lags = np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
    cov = np.outer(sd, sd) * rho**lags
    exact = -multivariate_normal(mean[0, :, 0].numpy(), cov).logpdf(x[0, :, 0].numpy())
    assert nll + 3 * np.log(2 * np.pi) == pytest.approx(exact, rel=1e-4)


def test_ar_noise_learns_persistence_where_it_exists():
    """Feature 0 is AR(1) with rho=0.7, feature 1 is white noise."""
    torch.manual_seed(0)
    n, t = 512, 20
    eps = torch.randn(n, t, 2)
    data = eps.clone()
    for k in range(1, t):
        data[:, k, 0] = 0.7 * data[:, k - 1, 0] + (1 - 0.49) ** 0.5 * eps[:, k, 0]
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    vae = VAECopula(input_dim=2, hidden_dim=16, latent_dim=2, seq_length=t, num_layers=1)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    vae.training_step(loader, opt, epochs=40, kl_annealing=False)
    vae.fit_copula(loader)
    rho = vae.decoder.noise_rho().detach()
    assert rho[0] > 0.3 and abs(rho[1]) < 0.2

    samples = vae.generate(512)
    x = samples - samples.mean(dim=1, keepdim=True)
    acf1 = ((x[:, 1:] * x[:, :-1]).sum(dim=(0, 1)) / (x * x).sum(dim=(0, 1))).numpy()
    assert acf1[0] > 0.45  # real within-window lag-1 ACF of an AR(0.7) is ~0.6
    assert abs(acf1[1]) < 0.2


def test_ar_noise_disabled_gives_zero_rho():
    vae = VAECopula(input_dim=3, hidden_dim=8, latent_dim=2, seq_length=5, ar_noise=False)
    assert torch.equal(vae.decoder.noise_rho(), torch.zeros(3))
    with pytest.raises(ValueError):
        Decoder(latent_dim=2, output_dim=3, ar_noise=True)


def test_gaussian_vae_generates_noise_not_smooth_curves():
    """Regression: a mean-only decoder produced smooth, near-flat samples on returns.

    Trained on white noise, the Gaussian VAE must reproduce the noise level
    and leave little step-to-step autocorrelation.
    """
    torch.manual_seed(0)
    data = torch.randn(256, 20, 2)
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    vae = VAECopula(input_dim=2, hidden_dim=16, latent_dim=4, seq_length=20, num_layers=1)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    vae.training_step(loader, opt, epochs=30, kl_annealing=False)
    vae.fit_copula(loader)
    samples = vae.generate(256)
    assert 0.6 < samples.std().item() < 1.5
    x = samples - samples.mean(dim=1, keepdim=True)
    acf1 = ((x[:, 1:] * x[:, :-1]).sum() / (x * x).sum()).item()
    assert abs(acf1) < 0.3


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
