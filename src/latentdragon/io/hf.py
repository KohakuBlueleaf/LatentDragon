"""Hugging Face exports: one folder for `transformers`, `diffusers` and LatentDragon.

The folder holds `config.json` (the `ARCH` plus summary fields), `model.safetensors` (the
`Autoencoder` state dict, unprefixed) and the remote code from `hf_remote/`. Both libraries'
`AutoModel.from_pretrained(folder, trust_remote_code=True)` build `LatentDragonModel` from it
with only torch and transformers installed; `from_pretrained` here rebuilds the full
LatentDragon `Autoencoder` from the same files.
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latentdragon.io.arch_json import arch_to_json, upgrade_arch
from latentdragon.io.state_keys import upgrade_keys
from latentdragon.model import build_autoencoder

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

REMOTE = Path(__file__).with_name("hf_remote")
REMOTE_FILES = ("configuration_latentdragon.py", "modeling_latentdragon.py")
CONFIG = "config.json"
WEIGHTS = "model.safetensors"
MODEL_TYPE = "latentdragon"


def hf_config(arch: dict) -> dict:
    """`config.json` of an export: the arch, the auto classes and the summary fields."""
    with torch.device("meta"):
        model = build_autoencoder(arch)
    config = {
        "model_type": MODEL_TYPE,
        "architectures": ["LatentDragonModel"],
        "auto_map": {
            "AutoConfig": "configuration_latentdragon.LatentDragonConfig",
            "AutoModel": "modeling_latentdragon.LatentDragonModel",
        },
        "arch": arch_to_json(arch),
        "latent_channels": arch["decoder"]["in_channels"],
        "spatial_scale": getattr(model.encoder, "spatial_scale", 1),
    }

    # Codebook layout, for readers that skip the arch
    quantizer = getattr(model.latent, "quantizer", None)
    if quantizer is not None:
        config.update(
            partitions=quantizer.partitions,
            partition_channels=quantizer.e_dim,
            codebook_size=quantizer.n_embed,
            rvq_levels=quantizer.rvq_levels,
        )
        if hasattr(model.latent, "eval_variants"):
            sizes = [ch // quantizer.e_dim for ch in model.latent.eval_variants()]
            config["nested_partitions"] = sizes
    return config


def save_pretrained(
    folder: str | Path, arch: dict, state: dict, card: str | None = None
) -> None:
    """Write the folder; `card` becomes its `README.md`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().clone().contiguous() for k, v in state.items()}
    save_file(tensors, str(folder / WEIGHTS), metadata={"format": "pt"})
    text = json.dumps(hf_config(arch), indent=2)
    (folder / CONFIG).write_text(text + "\n", encoding="utf-8")
    for name in REMOTE_FILES:
        shutil.copy(REMOTE / name, folder / name)
    if card is not None:
        (folder / "README.md").write_text(card, encoding="utf-8")


def resolve_folder(source: str | Path, revision: str | None = None) -> Path:
    """A local folder, or a Hub repo id downloaded to the cache (config and weights only)."""
    path = Path(source)
    if path.is_dir():
        return path
    if snapshot_download is None:
        raise ImportError(
            f"{source} is not a folder; install huggingface_hub to download it"
        )
    local = snapshot_download(
        repo_id=str(source), revision=revision, allow_patterns=[CONFIG, WEIGHTS]
    )
    return Path(local)


def from_pretrained(source: str | Path, revision: str | None = None):
    """Strictly rebuild an `Autoencoder` from an export folder or Hub repo id."""
    folder = resolve_folder(source, revision)
    config = json.loads((folder / CONFIG).read_text("utf-8"))
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"{source} is not a LatentDragon export")
    model = build_autoencoder(upgrade_arch(config["arch"]))
    state = upgrade_keys(load_file(str(folder / WEIGHTS)))
    model.load_state_dict(state, strict=True)
    return model
