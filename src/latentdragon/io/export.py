"""Exports: weights plus a rebuildable spec, and a diffusers-style folder."""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latentdragon.io import diffusers_model
from latentdragon.io.arch_json import arch_to_json, upgrade_arch
from latentdragon.io.state_keys import upgrade_keys
from latentdragon.model import build_autoencoder

CONFIG = "config.json"
WEIGHTS = "diffusion_pytorch_model.safetensors"


def _cpu(state: dict) -> dict:
    return {k: v.detach().cpu().clone().contiguous() for k, v in state.items()}


def save_export(folder: str | Path, arch: dict, state: dict) -> None:
    """`model.safetensors` and `spec.json`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(_cpu(state), str(folder / "model.safetensors"))
    text = json.dumps(arch_to_json(arch), indent=2)
    (folder / "spec.json").write_text(text, encoding="utf-8")


def load_export(folder: str | Path):
    """Strictly rebuild an `Autoencoder` from `save_export` output, older ones included."""
    folder = Path(folder)
    arch = upgrade_arch(json.loads((folder / "spec.json").read_text("utf-8")))
    model = build_autoencoder(arch)
    state = upgrade_keys(load_file(str(folder / "model.safetensors")))
    model.load_state_dict(state, strict=True)
    return model


def scale_levels(arch: dict) -> list[int]:
    """`block_out_channels` for a diffusers caller, which reads only its length.

    One entry per power of two of the encoder's declared spatial scale, plus one.
    """
    with torch.device("meta"):
        encoder = build_autoencoder(arch).encoder
    scale = getattr(encoder, "spatial_scale", 1)
    return [arch["decoder"]["in_channels"]] * scale.bit_length()


def diffusers_config(arch: dict, mean=None, std=None) -> dict:
    """Keys a diffusers `AutoModel` consumer reads, with the rebuildable arch inside.

    `latent_channels` is the decoder's input width; diffusers callers read one latent tensor.
    """
    channels = arch["decoder"]["in_channels"]
    return {
        "_class_name": "LatentDragonVAE",
        "auto_map": {"AutoModel": "diffusers_model.LatentDragonVAE"},
        "latent_channels": channels,
        "block_out_channels": scale_levels(arch),
        "latents_mean": list(mean) if mean is not None else [0.0] * channels,
        "latents_std": list(std) if std is not None else [1.0] * channels,
        "arch": arch_to_json(arch),
    }


def save_diffusers(
    folder: str | Path, arch: dict, state: dict, mean=None, std=None
) -> None:
    """A folder `AutoModel.from_pretrained(trust_remote_code=True)` can load."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    wrapped = {f"model.{k}": v for k, v in _cpu(state).items()}
    save_file(wrapped, str(folder / WEIGHTS))
    text = json.dumps(diffusers_config(arch, mean, std), indent=2)
    (folder / CONFIG).write_text(text, encoding="utf-8")
    shutil.copy(diffusers_model.__file__, folder / "diffusers_model.py")
