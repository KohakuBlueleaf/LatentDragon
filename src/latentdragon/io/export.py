"""Exports: weights plus a rebuildable spec, and a diffusers-style folder."""

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from latentdragon.io import diffusers_model
from latentdragon.io.arch_json import arch_from_json, arch_to_json, diffusers_levels
from latentdragon.model import AEArchConfig, build_autoencoder

CONFIG = "config.json"
WEIGHTS = "diffusion_pytorch_model.safetensors"


def _cpu(state: dict) -> dict:
    return {k: v.detach().cpu().clone().contiguous() for k, v in state.items()}


def save_export(folder: str | Path, arch: AEArchConfig, state: dict) -> None:
    """`model.safetensors` and `spec.json`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(_cpu(state), str(folder / "model.safetensors"))
    text = json.dumps(arch_to_json(arch), indent=2)
    (folder / "spec.json").write_text(text, encoding="utf-8")


def load_export(folder: str | Path):
    """Strictly rebuild an `Autoencoder` from `save_export` output."""
    folder = Path(folder)
    arch = arch_from_json(json.loads((folder / "spec.json").read_text("utf-8")))
    model = build_autoencoder(arch)
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    return model


def diffusers_config(arch: AEArchConfig, mean=None, std=None) -> dict:
    """Keys a diffusers `AutoModel` consumer reads, with the rebuildable arch inside."""
    channels = arch.latent_dim
    return {
        "_class_name": "LatentDragonVAE",
        "auto_map": {"AutoModel": "diffusers_model.LatentDragonVAE"},
        "latent_channels": channels,
        "block_out_channels": diffusers_levels(arch),
        "latents_mean": list(mean) if mean is not None else [0.0] * channels,
        "latents_std": list(std) if std is not None else [1.0] * channels,
        "arch": arch_to_json(arch),
    }


def save_diffusers(
    folder: str | Path, arch: AEArchConfig, state: dict, mean=None, std=None
) -> None:
    """A folder `AutoModel.from_pretrained(trust_remote_code=True)` can load."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    wrapped = {f"model.{k}": v for k, v in _cpu(state).items()}
    save_file(wrapped, str(folder / WEIGHTS))
    text = json.dumps(diffusers_config(arch, mean, std), indent=2)
    (folder / CONFIG).write_text(text, encoding="utf-8")
    shutil.copy(diffusers_model.__file__, folder / "diffusers_model.py")
