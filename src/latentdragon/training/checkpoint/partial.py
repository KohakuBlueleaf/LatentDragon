"""`INIT_FROM`: load matching weights into a fresh run, with a report."""

from pathlib import Path

import torch
from safetensors.torch import load_file

from latentdragon.io.state_keys import upgrade_keys
from latentdragon_common.log import get_logger

log = get_logger(__name__)


def checkpoint_weights(path: str | Path, use_ema: bool = True) -> dict:
    """Model tensors of a safetensors file or checkpoint folder, EMA laid on top."""
    path = Path(path)
    if not path.is_dir():
        return upgrade_keys(load_file(str(path)))
    tensors = {
        k.removeprefix("model."): v
        for k, v in load_file(str(path / "trainable.safetensors")).items()
        if k.startswith("model.")
    }
    ema = path / "ema.safetensors"
    if use_ema and ema.exists():
        tensors.update(load_file(str(ema)))
    return upgrade_keys(tensors)


def split_for_module(
    module: torch.nn.Module, tensors: dict[str, torch.Tensor]
) -> tuple[dict, list[str]]:
    """Tensors matching `module` by name and shape, plus the skipped names."""
    live = module.state_dict()
    matched, skipped = {}, []
    for key, value in tensors.items():
        name = key.removeprefix("model.")
        if name in live and live[name].shape == value.shape:
            matched[name] = value
        else:
            skipped.append(key)
    return matched, skipped


def load_partial(module: torch.nn.Module, path: str | Path) -> dict[str, int]:
    """Load what matches; returns loaded, skipped and missing counts."""
    matched, skipped = split_for_module(module, checkpoint_weights(path))
    module.load_state_dict(matched, strict=False)
    missing = sorted(set(module.state_dict()) - set(matched))
    report = {"loaded": len(matched), "skipped": len(skipped), "missing": len(missing)}
    log.info("init_from", path=str(path), **report)
    return {**report, "missing_keys": missing}


def refuse_frozen_missing(report: dict, frozen: list[str]) -> None:
    """A frozen part must have come from the weights; random frozen state is an error."""
    bad = [k for k in report["missing_keys"] if k.startswith(tuple(frozen))]
    if bad:
        raise ValueError(f"INIT_FROM left frozen weights unset: {bad[:5]}")
