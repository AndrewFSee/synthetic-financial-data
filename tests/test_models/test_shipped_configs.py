"""Every shipped model config must build and run at realistic dimensions.

The unit tests mostly use tiny architectures; this guards the configs users
actually run (the default 5-level U-Net once crashed on its first step).
"""

import pytest
import torch

from synfin.models.diffusion.unet import UNet1D
from synfin.models.factory import MODEL_NAMES, create_model, model_kwargs_from_config
from synfin.utils.config import load_config

SEQ, FEAT = 30, 5


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_shipped_config_forward_backward(name):
    cfg = load_config(f"configs/{name}.yaml")
    model = create_model(name, model_kwargs_from_config(name, FEAT, SEQ, cfg["model"]))
    x = torch.randn(4, SEQ, FEAT)
    if name == "timegan":
        loss = (model.recover(model.embed(x)) - x).pow(2).mean()
    elif name == "vae_copula":
        loss = model.negative_elbo(x)[0]
    else:
        loss = model(x)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "hidden_dims,num_res_blocks",
    [
        ([64, 128, 256, 128, 64], 2),  # shipped default
        ([16, 32, 16], 1),
        ([16, 32, 16], 3),
        ([8, 16, 32, 64, 32, 16, 8], 2),
    ],
)
def test_unet_skip_routing(hidden_dims, num_res_blocks):
    net = UNet1D(in_channels=FEAT, hidden_dims=hidden_dims, num_res_blocks=num_res_blocks)
    out = net(torch.randn(3, SEQ, FEAT), torch.randint(0, 100, (3,)))
    assert out.shape == (3, SEQ, FEAT)
