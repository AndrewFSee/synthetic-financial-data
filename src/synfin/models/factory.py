"""Build models from config sections, and rebuild them from checkpoint kwargs.

Training resolves a ``model`` config section into explicit constructor kwargs
(:func:`model_kwargs_from_config`), which are stored in the checkpoint;
generation rebuilds the identical model with :func:`create_model`.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

MODEL_NAMES = ("timegan", "diffusion", "diffusion_ts", "vae_copula")


def model_kwargs_from_config(
    model_name: str,
    input_dim: int,
    seq_length: int,
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate a ``model`` config section into constructor kwargs.

    Args:
        model_name: One of :data:`MODEL_NAMES`.
        input_dim: Number of features (taken from the data, not the config).
        seq_length: Window length (taken from the data).
        model_cfg: The ``model`` section of the merged config.

    Returns:
        Keyword arguments for the model class.
    """
    c = model_cfg
    if model_name == "timegan":
        return {
            "input_dim": input_dim,
            "hidden_dim": c.get("hidden_dim", 24),
            "num_layers": c.get("num_layers", 3),
            "noise_dim": c.get("noise_dim", c.get("hidden_dim", 24)),
            "rnn_type": c.get("rnn_type", "gru"),
            "dropout": c.get("dropout", 0.0),
        }
    if model_name == "diffusion":
        unet = c.get("unet", {})
        return {
            "in_channels": input_dim,
            "seq_length": seq_length,
            "num_timesteps": c.get("num_timesteps", 1000),
            "noise_schedule": c.get("noise_schedule", "cosine"),
            "beta_start": c.get("beta_start", 1e-4),
            "beta_end": c.get("beta_end", 0.02),
            "hidden_dims": list(unet.get("hidden_dims", [64, 128, 256, 128, 64])),
            "time_embed_dim": unet.get("time_embed_dim", 128),
            "num_res_blocks": unet.get("num_res_blocks", 2),
            "dropout": unet.get("dropout", 0.1),
            "groups": unet.get("group_norm_groups", 8),
        }
    if model_name == "diffusion_ts":
        return {
            "in_channels": input_dim,
            "seq_length": seq_length,
            "num_timesteps": c.get("num_timesteps", 1000),
            "noise_schedule": c.get("noise_schedule", "cosine"),
            "beta_start": c.get("beta_start", 1e-4),
            "beta_end": c.get("beta_end", 0.02),
            "d_model": c.get("d_model", 128),
            "n_heads": c.get("n_heads", 8),
            "n_layers": c.get("n_layers", 3),
            "dim_feedforward": c.get("dim_feedforward", 256),
            "dropout": c.get("dropout", 0.1),
            "time_embed_dim": c.get("time_embed_dim", 128),
            "trend_degree": c.get("trend_degree", 3),
            "num_harmonics": c.get("num_harmonics", 6),
            "fourier_loss_weight": c.get("fourier_loss_weight", 0.1),
        }
    if model_name == "vae_copula":
        copula = c.get("copula", {})
        return {
            "input_dim": input_dim,
            "hidden_dim": c.get("hidden_dim", 128),
            "latent_dim": c.get("latent_dim", 32),
            "seq_length": seq_length,
            "num_layers": c.get("num_layers", 2),
            "rnn_type": c.get("rnn_type", "lstm"),
            "dropout": c.get("dropout", 0.1),
            "recon_loss": c.get("reconstruction_loss", "gaussian"),
            "ar_noise": c.get("ar_noise", True),
            "noise": c.get("noise_distribution", "gaussian"),
            "garch_noise": c.get("garch_noise", False),
            "copula_type": copula.get("type", "gaussian"),
            "copula_df": float(copula.get("df", 4.0)),
        }
    raise ValueError(f"Unknown model {model_name!r}; use one of {MODEL_NAMES}.")


def create_model(model_name: str, kwargs: Dict[str, Any]) -> nn.Module:
    """Instantiate a model by name with explicit kwargs."""
    if model_name == "timegan":
        from synfin.models.timegan import TimeGAN

        return TimeGAN(**kwargs)
    if model_name == "diffusion":
        from synfin.models.diffusion import DiffusionModel

        return DiffusionModel(**kwargs)
    if model_name == "diffusion_ts":
        from synfin.models.diffusion_ts import DiffusionTS

        return DiffusionTS(**kwargs)
    if model_name == "vae_copula":
        from synfin.models.vae_copula import VAECopula

        return VAECopula(**kwargs)
    raise ValueError(f"Unknown model {model_name!r}; use one of {MODEL_NAMES}.")
