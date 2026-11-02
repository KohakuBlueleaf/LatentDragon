"""Loading: every arch file builds, and every file format rebuilds the same model."""

import torch
from _tiny import ROOT, run_checks
from kohakuengine import load_config_file

from latentdragon.io.arch_json import arch_from_json
from latentdragon.io.diffusers_model import LatentDragonVAE
from latentdragon.io.export import load_export, save_diffusers, save_export
from latentdragon.io.legacy import (
    read_lightning,
    read_safetensors,
    write_lightning,
    write_safetensors,
)
from latentdragon.model import AEArchConfig, build_autoencoder

TINY = {
    "latent_dim": 4,
    "encoder": "conv",
    "encoder_blocks": ((2, 32, 1),),
    "encoder_attn": None,
    "trunk": "conv",
    "trunk_blocks": ((2, 32, 1),),
    "trunk_attn": None,
    "trunk_norm": "group32",
    "bottleneck": {
        "name": "vq",
        "quantizer": {"name": "kohaku", "n_embed": 8, "partition_mode": "hyper"},
    },
}


def tiny_model():
    """A small KohakuVQ model and its arch."""
    arch = AEArchConfig(**TINY)
    return arch, build_autoencoder(arch).eval()


def rebuilt(arch, state: dict):
    """A new model holding `state`; a missing or extra key raises."""
    model = build_autoencoder(arch)
    model.load_state_dict(state, strict=True)
    return model.eval()


def assert_same_output(first, second) -> None:
    x = torch.randn(1, 3, 8, 8)
    assert torch.equal(first(x)[0], second(x)[0])


def every_arch_file_builds(folder) -> None:
    files = sorted((ROOT / "configs" / "ae" / "arch").glob("*.py"))
    assert files
    with torch.device("meta"):
        for path in files:
            build_autoencoder(
                arch_from_json(load_config_file(path).globals_dict["ARCH"])
            )


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
    expected = model.bottleneck(model.encode(x)).raw
    assert torch.equal(loaded.encode(x).latent_dist.mean, expected)


if __name__ == "__main__":
    run_checks(
        every_arch_file_builds,
        native_export_loads,
        legacy_safetensors_loads,
        legacy_lightning_loads,
        diffusers_folder_loads,
    )
