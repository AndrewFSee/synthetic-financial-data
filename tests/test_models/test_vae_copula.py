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


def test_student_t_nll_matches_scipy_density():
    """Without AR, the NLL is the standardized-t log density (constant-matched)."""
    from scipy.stats import t as student_t

    torch.manual_seed(1)
    dec = Decoder(
        latent_dim=3,
        hidden_dim=8,
        output_dim=2,
        seq_length=7,
        num_layers=1,
        heteroscedastic=True,
        noise="student_t",
    )
    z, x = torch.randn(4, 3), torch.randn(4, 7, 2) * 2
    with torch.no_grad():
        total = dec.nll(x, z).item() * x.numel()
        mean, log_var = dec.forward_dist(z)
        df = dec.noise_df().numpy()
    sd = np.exp(0.5 * log_var.numpy())
    c = np.sqrt((df - 2) / df)
    e = (x.numpy() - mean.numpy()) / sd
    exact = -(student_t.logpdf(e / c, df) - np.log(c) - np.log(sd)).sum()
    assert total + 0.5 * np.log(2 * np.pi) * x.numel() == pytest.approx(exact, rel=1e-5)


def test_student_t_noise_learns_fat_tails_where_they_exist():
    """Feature 0 has t(3) shocks, feature 1 Gaussian: df(0) ends well below df(1)."""
    torch.manual_seed(0)
    n, t = 512, 20
    data = torch.randn(n, t, 2)
    fat = torch.distributions.StudentT(torch.tensor(3.0)).sample((n, t))
    data[:, :, 0] = fat / np.sqrt(3.0)  # unit variance
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    vae = VAECopula(
        input_dim=2, hidden_dim=16, latent_dim=2, seq_length=t, num_layers=1, noise="student_t"
    )
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    vae.training_step(loader, opt, epochs=80, kl_annealing=False)
    df = vae.decoder.noise_df().detach()
    assert df[0] < 5.0 < df[1]
    vae.fit_copula(loader)
    s = vae.generate(1024)
    tail = lambda v: float(v.abs().quantile(0.999) / v.std())  # noqa: E731
    assert tail(s[:, :, 0]) > 4.5 > tail(s[:, :, 1])  # Gaussian ~3.3, unit t(3) ~7.4


def test_garch_nll_matches_reference_recursion():
    """GARCH(1,1) likelihood (Gaussian innovations, no AR) against a direct numpy version."""
    torch.manual_seed(2)
    dec = Decoder(
        latent_dim=3,
        hidden_dim=8,
        output_dim=1,
        seq_length=8,
        num_layers=1,
        heteroscedastic=True,
        garch_noise=True,
    )
    z, x = torch.randn(2, 3), torch.randn(2, 8, 1) * 1.5
    with torch.no_grad():
        total = dec.nll(x, z).item() * x.numel()
        mean, log_var = dec.forward_dist(z)
        alpha, beta = (p.item() for p in dec.garch_params())
    e = ((x - mean) * torch.exp(-0.5 * log_var)).numpy()[:, :, 0]
    lv = log_var.numpy()[:, :, 0]
    expected = 0.0
    for b in range(e.shape[0]):
        g = 1.0
        for t in range(e.shape[1]):
            if t > 0:
                g = (1 - alpha - beta) + alpha * e[b, t - 1] ** 2 + beta * g
            expected += 0.5 * lv[b, t] + 0.5 * np.log(g) + 0.5 * e[b, t] ** 2 / g
    assert total == pytest.approx(expected, rel=1e-5)


def test_garch_noise_learns_volatility_feedback():
    """On GARCH data the learned alpha grows and samples show |x| autocorrelation."""
    rng = np.random.default_rng(0)
    n, t, a, b = 512, 30, 0.2, 0.75
    data = np.empty((n, t, 1))
    for i in range(n):
        g, x_prev = 1.0, 0.0
        for k in range(t):
            if k > 0:
                g = (1 - a - b) + a * x_prev**2 + b * g
            x_prev = np.sqrt(g) * rng.standard_normal()
            data[i, k, 0] = x_prev
    loader = torch.utils.data.DataLoader(
        torch.tensor(data, dtype=torch.float32), batch_size=64, shuffle=True
    )
    torch.manual_seed(0)
    vae = VAECopula(
        input_dim=1, hidden_dim=16, latent_dim=2, seq_length=t, num_layers=1, garch_noise=True
    )
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    vae.training_step(loader, opt, epochs=40, kl_annealing=False)
    alpha, _ = vae.decoder.garch_params()
    assert alpha.item() > 0.1  # started at 0.05
    vae.fit_copula(loader)
    s = vae.generate(512)[:, :, 0].abs()
    s = s - s.mean(dim=1, keepdim=True)
    assert ((s[:, 1:] * s[:, :-1]).sum() / (s * s).sum()).item() > 0.05


def _fitted_small_vae():
    torch.manual_seed(0)
    data = torch.randn(96, 10, 3)
    loader = torch.utils.data.DataLoader(data, batch_size=32)
    vae = VAECopula(input_dim=3, hidden_dim=8, latent_dim=4, seq_length=10, num_layers=1)
    vae.training_step(loader, torch.optim.Adam(vae.parameters(), lr=1e-2), epochs=3)
    vae.fit_latent_sampler(loader)
    return vae, data


def test_posterior_sampler_stores_and_samples_aggregate_posterior():
    vae, data = _fitted_small_vae()
    assert vae.posterior_mu.shape == (96, 4)
    with torch.no_grad():
        mu, log_var = vae.encoder(data)
    assert torch.allclose(vae.posterior_mu, mu)
    # Latents sampled with temperature 0 are exactly stored posterior means.
    torch.manual_seed(1)
    out = vae.generate(16, sampler="posterior", temperature=0.0)
    assert out.shape == (16, 10, 3)
    for name in ("posterior", "copula", "prior", "auto"):
        assert vae.generate(4, sampler=name).shape == (4, 10, 3)
    with pytest.raises(ValueError):
        vae.generate(4, sampler="nope")


def test_posterior_buffers_survive_checkpoint_roundtrip(tmp_path):
    from synfin.training.checkpoint import load_checkpoint, save_checkpoint

    vae, _ = _fitted_small_vae()
    kwargs = {"input_dim": 3, "hidden_dim": 8, "latent_dim": 4, "seq_length": 10, "num_layers": 1}
    path = save_checkpoint(tmp_path / "v.pt", vae, model_name="vae_copula", model_kwargs=kwargs)
    loaded, _ = load_checkpoint(path)  # fresh model has empty (0, 4) buffers
    assert torch.equal(loaded.posterior_mu, vae.posterior_mu)
    assert torch.equal(loaded.posterior_log_var, vae.posterior_log_var)
    assert loaded.generate(4).shape == (4, 10, 3)


def test_unfitted_sampler_errors_and_auto_falls_back():
    vae = VAECopula(input_dim=3, hidden_dim=8, latent_dim=4, seq_length=10, num_layers=1)
    with pytest.raises(RuntimeError):
        vae.generate(2, sampler="posterior")
    with pytest.raises(RuntimeError):
        vae.generate(2, sampler="copula")
    assert vae.generate(2).shape == (2, 10, 3)  # auto -> prior


def test_noise_option_validation():
    with pytest.raises(ValueError):
        Decoder(latent_dim=2, output_dim=3, noise="laplace", heteroscedastic=True)
    with pytest.raises(ValueError):
        Decoder(latent_dim=2, output_dim=3, noise="student_t")


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
