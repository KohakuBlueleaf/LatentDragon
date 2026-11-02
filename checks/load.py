"""Loading: every arch file builds, every file format rebuilds the same model, and the
flat arch and part names of older runs and exports still load."""

import json

import torch
from _tiny import ROOT, run_checks
from kohakuengine import load_config_file
from safetensors.torch import save_file

from latentdragon.io.arch_json import flat_fields, upgrade_arch
from latentdragon.io.diffusers_model import LatentDragonVAE
from latentdragon.io.export import load_export, save_diffusers, save_export
from latentdragon.io.legacy import (
    read_lightning,
    read_safetensors,
    write_lightning,
    write_safetensors,
)
from latentdragon.model import build_autoencoder

# The flat form older runs and exports wrote; `upgrade_arch` reads it
FLAT_TINY = {
    "latent_dim": 4,
    "encoder": "conv",
    "encoder_blocks": [[2, 32, 1]],
    "encoder_attn": None,
    "encoder_norm": "rms",
    "trunk": "conv",
    "trunk_blocks": [[2, 32, 1]],
    "trunk_attn": None,
    "trunk_norm": "group32",
    "bottleneck": {
        "name": "vq",
        "quantizer": {"name": "kohaku", "n_embed": 8, "partition_mode": "hyper"},
    },
}
OLD_PARTS = (("latent.", "bottleneck."), ("decoder.", "trunk."))


def tiny_model():
    """A small KohakuVQ model and its arch."""
    arch = upgrade_arch(FLAT_TINY)
    return arch, build_autoencoder(arch).eval()


def rebuilt(arch, state: dict):
    """A new model holding `state`; a missing or extra key raises."""
    model = build_autoencoder(arch)
    model.load_state_dict(state, strict=True)
    return model.eval()


def assert_same_output(first, second) -> None:
    x = {"x": torch.randn(1, 3, 8, 8)}
    assert torch.equal(first(x)[0]["x"], second(x)[0]["x"])


def every_arch_file_builds(folder) -> None:
    files = sorted((ROOT / "configs" / "ae" / "arch").rglob("*.py"))
    assert files
    with torch.device("meta"):
        for path in files:
            build_autoencoder(load_config_file(path).globals_dict["ARCH"])


def flat_arch_round_trips(folder) -> None:
    arch, _ = tiny_model()
    assert upgrade_arch(flat_fields(arch)) == arch


def old_export_loads(folder) -> None:
    arch, model = tiny_model()
    state = {}
    for key, value in model.state_dict().items():
        for new, old in OLD_PARTS:
            key = old + key.removeprefix(new) if key.startswith(new) else key
        state[key] = value.contiguous()
    save_file(state, str(folder / "model.safetensors"))
    (folder / "spec.json").write_text(json.dumps(FLAT_TINY), encoding="utf-8")
    assert_same_output(model, load_export(folder).eval())


def native_export_loads(folder) -> None:
    arch, model = tiny_model()
    save_export(folder, arch, model.state_dict())
    assert_same_output(model, load_export(folder).eval())


def legacy_safetensors_loads(folder) -> None:
    arch, model = tiny_model()
    path = folder / "model.safetensors"
    write_safetensors(path, arch, model.state_dict())
    assert_same_output(model, rebuilt(*read_safetensors(path)))


def legacy_lightning_loads(folder) -> None:
    arch, model = tiny_model()
    path = folder / "model.ckpt"
    write_lightning(path, arch, model.state_dict())
    read_arch, live, _ = read_lightning(path)
    assert_same_output(model, rebuilt(read_arch, live))


def diffusers_folder_loads(folder) -> None:
    arch, model = tiny_model()
    save_diffusers(folder, arch, model.state_dict())
    loaded = LatentDragonVAE.from_pretrained(folder).eval()

    x = torch.randn(1, 3, 8, 8)
    expected = model.latent.raw(model.encode({"x": x}))["z"]
    assert torch.equal(loaded.encode(x).latent_dist.mean, expected)


if __name__ == "__main__":
    run_checks(
        every_arch_file_builds,
        flat_arch_round_trips,
        old_export_loads,
        native_export_loads,
        legacy_safetensors_loads,
        legacy_lightning_loads,
        diffusers_folder_loads,
    )
