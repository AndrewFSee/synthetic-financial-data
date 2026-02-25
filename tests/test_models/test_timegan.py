"""Tests for TimeGAN model."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synfin.models.timegan import TimeGAN
from synfin.models.timegan.embedder import Embedder
from synfin.models.timegan.recovery import Recovery
from synfin.models.timegan.generator import Generator
from synfin.models.timegan.discriminator import Discriminator
from synfin.models.timegan.supervisor import Supervisor


@pytest.fixture
def timegan():
    return TimeGAN(input_dim=5, hidden_dim=16, num_layers=2)


def test_embedder_shape():
    """Embedder maps (B, T, C) -> (B, T, H)."""
    emb = Embedder(input_dim=5, hidden_dim=16)
    x = torch.randn(4, 10, 5)
    h = emb(x)
    assert h.shape == (4, 10, 16)


def test_recovery_shape():
    """Recovery maps (B, T, H) -> (B, T, C)."""
    rec = Recovery(hidden_dim=16, output_dim=5)
    h = torch.randn(4, 10, 16)
    x = rec(h)
    assert x.shape == (4, 10, 5)


def test_generator_shape():
    """Generator maps (B, T, noise_dim) -> (B, T, H)."""
    gen = Generator(noise_dim=16, hidden_dim=16)
    z = torch.randn(4, 10, 16)
    e_hat = gen(z)
    assert e_hat.shape == (4, 10, 16)


def test_discriminator_shape():
    """Discriminator maps (B, T, H) -> (B, T, 1)."""
    disc = Discriminator(hidden_dim=16)
    h = torch.randn(4, 10, 16)
    logits = disc(h)
    assert logits.shape == (4, 10, 1)


def test_supervisor_shape():
    """Supervisor maps (B, T, H) -> (B, T, H)."""
    sup = Supervisor(hidden_dim=16, num_layers=1)
    h = torch.randn(4, 10, 16)
    h_hat = sup(h)
    assert h_hat.shape == (4, 10, 16)


def test_timegan_generate_shape(timegan):
    """TimeGAN.generate produces correct output shape."""
    samples = timegan.generate(num_samples=8, seq_length=10)
    assert samples.shape == (8, 10, 5)


def test_timegan_embed_recover(timegan):
    """Embed then recover should produce same shape as input."""
    x = torch.randn(4, 10, 5)
    h = timegan.embed(x)
    x_hat = timegan.recover(h)
    assert x_hat.shape == x.shape


def test_timegan_lstm():
    """TimeGAN works with LSTM cells."""
    model = TimeGAN(input_dim=5, hidden_dim=16, num_layers=2, rnn_type="lstm")
    samples = model.generate(num_samples=4, seq_length=10)
    assert samples.shape == (4, 10, 5)
