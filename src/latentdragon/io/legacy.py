"""Read legacy checkpoints and write files in the legacy layout."""

import json
import pickle
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from latentdragon.io.legacy_hparams import from_legacy_hparams, to_legacy_hparams
from latentdragon.io.legacy_keys import from_legacy, split_lightning, to_legacy


class _Placeholder:
    """Stands in for pickled classes of packages that are not installed."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickle trusted checkpoints when `latentmaid.*` classes are absent."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Placeholder


class tolerant_pickle:
    """The `pickle_module` handed to `torch.load`."""

    Unpickler = _TolerantUnpickler
    load = staticmethod(pickle.load)


def _conv_decoder(arch: dict) -> bool:
    """Whether the decoder is the SD-VAE family, whose legacy head is `conv_out`."""
    decoder = arch["decoder"]
    return (decoder if isinstance(decoder, str) else decoder["name"]) == "sdvae"


def load_lightning(path: str | Path) -> dict:
    """Read a trusted Lightning `.ckpt`; pickled code runs, so only use your own."""
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=tolerant_pickle
    )


def read_lightning(path: str | Path) -> tuple[dict, dict, dict]:
    """`(arch, live_state, ema_state)` in the new layout."""
    checkpoint = load_lightning(path)
    live, ema = split_lightning(checkpoint["state_dict"])
    arch = from_legacy_hparams(checkpoint["hyper_parameters"])
    return arch, from_legacy(live), from_legacy(ema) if ema else {}


def read_safetensors(path: str | Path) -> tuple[dict, dict]:
    """`(arch, state)` from a legacy safetensors export."""
    with safe_open(str(path), framework="pt") as handle:
        metadata = json.loads(handle.metadata()["config"])
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    return from_legacy_hparams(metadata), from_legacy(state)


def write_safetensors(path: str | Path, arch: dict, state: dict) -> None:
    """A safetensors file the research `tokenizer` and `statedict_to_model` read."""
    hparams = to_legacy_hparams(arch)
    metadata = dict(hparams)
    metadata.pop("variational")
    metadata["enc_class"] = "latentmaid.model.Encoder"
    metadata["dec_class"] = "latentmaid.model.Decoder"
    legacy = to_legacy(_cpu(state), _conv_decoder(arch))
    save_file(legacy, str(path), metadata={"config": json.dumps(metadata)})


def write_lightning(path: str | Path, arch: dict, state: dict) -> None:
    """A minimal `.ckpt` for `LatentMaid.load_from_checkpoint(strict=False)`."""
    hparams = to_legacy_hparams(arch)
    legacy = to_legacy(_cpu(state), _conv_decoder(arch))
    weights = {
        ("vq_handler." + k if k.startswith("vq.") else k): v for k, v in legacy.items()
    }
    payload = {
        "state_dict": weights,
        "hyper_parameters": hparams,
        "pytorch-lightning_version": "2.6.6",
    }
    torch.save(payload, path)


def _cpu(state: dict) -> dict:
    return {k: v.detach().cpu().clone().contiguous() for k, v in state.items()}
