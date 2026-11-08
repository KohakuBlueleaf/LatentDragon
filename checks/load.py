"""Loading: every arch file builds, every file format rebuilds the same model, and the
flat arch and part names of older runs and exports still load."""

import json

import torch
from _tiny import ROOT, run_checks
from kohakuengine import load_config_file
from safetensors.torch import save_file
from transformers import AutoModel

from latentdragon.io.arch_json import flat_fields, upgrade_arch
from latentdragon.io.diffusers_model import LatentDragonVAE
from latentdragon.io.export import load_export, save_diffusers, save_export
from latentdragon.io.hf import from_pretrained, save_pretrained
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

# Everything the Hub remote code reimplements: LatentMaid, nested partitions, two levels
HF_TINY = {
    "encoder": {
        "name": "latentmaid",
        "out_channels": 8,
        "stages": [[2, 16, 16, 1]],
        "norm": "layer",
    },
    "latent": {
        "name": "nested",
        "dim": 8,
        "unit": "partition",
        "latent_chs": [1, 2],
        "inner": {
            "name": "vq",
            "quantizer": {
                "name": "kohaku",
                "n_embed": 16,
                "partitions": 2,
                "rvq_levels": 2,
                "partition_mode": "hyper",
                "separated_proj": False,
                "mlp_projector": True,
                "mlp_hidden_dim": 16,
                "base_dim": 4,
                "trainable_base": True,
                "dim_scale": True,
            },
        },
    },
    "decoder": {
        "name": "latentmaid",
        "in_channels": 8,
        "stages": [[2, 32, 16, 1]],
        "norm": "layer",
    },
    "head": {"name": "conv", "in_channels": 8, "fp32": True},
}


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


def bottleneck_is_the_original_layout(folder) -> None:
    """A bare encoder, a `bottleneck` and a decoder at its first-stage width compute the
    original LatentMaid model, given its encoder output and decoder input 1x1s; a nested def
    inside masks the latent before the up conv, and a `kl` inner gets twice the channels.
    """
    encoder = {"name": "latentmaid", "stages": [[2, 16, 16, 1]], "norm": "layer"}
    decoder = {"name": "latentmaid", "stages": [[2, 16, 16, 1]], "norm": "layer"}
    head = {"name": "conv", "in_channels": 4}
    torch.manual_seed(0)
    plain = build_autoencoder(
        {
            "encoder": {**encoder, "out_channels": 4},
            "latent": "identity",
            "decoder": {**decoder, "in_channels": 4},
            "head": head,
        }
    ).double()
    split_arch = {
        "encoder": {**encoder, "out_channels": None},
        "latent": {
            "name": "bottleneck",
            "in_channels": 16,
            "dim": 4,
            "out_channels": 16,
            "inner": {"name": "nested", "latent_chs": [2]},
        },
        "decoder": {**decoder, "in_channels": 16},
        "head": head,
    }
    split = build_autoencoder(split_arch).double().eval()
    state = plain.state_dict()
    for old, new in (
        ("encoder.out_proj", "latent.down"),
        ("decoder.stage_projs.0", "latent.up"),
    ):
        for kind in ("weight", "bias"):
            state[f"{new}.{kind}"] = state.pop(f"{old}.{kind}")
    split.load_state_dict(state, strict=True)
    assert isinstance(split.decoder.stage_projs[0], torch.nn.Identity)
    assert split.encoder.out_channels == 16
    assert split.encoder.last_layer is split.encoder.stages[-1][-1].writer
    x = {"x": torch.randn(2, 3, 8, 8, dtype=torch.float64)}
    assert torch.equal(plain.eval()(x)[0]["x"], split(x)[0]["x"])

    # The variant masks the 4-channel latent, then the up conv reads it
    with torch.no_grad():
        latent = split.encode(x).latent
        half = split.latent.variant(latent, 2)
        z = latent["z"].clone()
        z[:, 2:] = 0
        assert torch.equal(split.latent.deserialize(half)["z"], split.latent.up(z))

    kl = build_autoencoder(
        {**split_arch, "latent": {**split_arch["latent"], "inner": "kl"}}
    ).latent
    assert kl.down.out_channels == 8 and kl.up.in_channels == 4
    assert torch.equal(kl.down.weight[4:], torch.zeros_like(kl.down.weight[4:]))


def hf_folder_loads(folder) -> None:
    """LatentDragon and the remote code read one Hub folder and agree with the source model."""
    torch.manual_seed(0)
    model = build_autoencoder(upgrade_arch(HF_TINY)).double().eval()
    save_pretrained(folder, HF_TINY, model.state_dict())
    x = torch.randn(2, 3, 16, 16, dtype=torch.float64)

    # LatentDragon rebuilds the same model
    loaded = from_pretrained(folder).double().eval()
    assert torch.equal(model({"x": x})[0]["x"], loaded({"x": x})[0]["x"])

    # The remote code: same indices, and the same image from every partition count
    remote = AutoModel.from_pretrained(folder, trust_remote_code=True)
    remote = remote.double().eval()
    with torch.no_grad():
        encoded = model.encode({"x": x})
        quantizer = model.latent.quantizer
        local = encoded.latent["z"]["indices"] - quantizer.index_offsets
        out = remote.encode(x)
        assert torch.equal(out.indices, local), "indices differ"
        for channels in model.latent.eval_variants():
            latent = model.latent.variant(encoded.latent, channels)
            expected = model.head(model.decode(latent))["x"]
            parts = channels // quantizer.e_dim
            got = remote.decode_indices(out.indices, parts).sample
            error = ((got - expected).abs().max() / expected.abs().max()).item()
            assert error < 1e-12, (parts, error)


if __name__ == "__main__":
    run_checks(
        every_arch_file_builds,
        flat_arch_round_trips,
        old_export_loads,
        native_export_loads,
        legacy_safetensors_loads,
        legacy_lightning_loads,
        diffusers_folder_loads,
        bottleneck_is_the_original_layout,
        hf_folder_loads,
    )
