"""Key maps between legacy checkpoints and the Autoencoder layout."""

import torch

HEAD_RULES = (
    ("decoder.out_proj.", "head.proj."),
    ("decoder.conv_out.", "head.proj."),
    ("decoder.norm_out.", "head.norm."),
    ("decoder.", "trunk."),
)
UNPORTED_VQ = ("codebook_mix_", "codebook_base_a", "codebook_base_b", "codebook_proj.")
LIGHTNING_DROPPED = ("loss.", "discriminator.", "ema.model.", "vq_handler._usage")
EMA_PREFIX = "ema.ema__m"
EMA_TARGETS = {"0": "encoder.", "1": "decoder.", "2": "vq."}


def model_key(key: str) -> str | None:
    """A legacy `encoder.`, `decoder.` or `vq.` key in the new layout, else None."""
    if key.startswith("encoder."):
        return key
    if key.startswith("vq."):
        return "bottleneck.vq.codebook." + key.removeprefix("vq.")
    for old, new in HEAD_RULES:
        if key.startswith(old):
            return new + key.removeprefix(old)
    return None


def check_vq_key(key: str) -> None:
    """Refuse weights of features that were not ported."""
    if key.startswith("vq.") and key.removeprefix("vq.").startswith(UNPORTED_VQ):
        raise ValueError(f"{key} belongs to a feature that is not ported")


def from_legacy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Plain legacy state (prefixes `encoder.`, `decoder.`, `vq.`) to the new layout."""
    out = {}
    for key, value in state.items():
        check_vq_key(key)
        new = model_key(key)
        if new is None:
            raise KeyError(f"unknown legacy key {key}")
        out[new] = value
    return out


def legacy_key(key: str, conv_trunk: bool) -> str:
    """A new-layout key in the legacy `encoder.`, `decoder.`, `vq.` layout."""
    if key.startswith("encoder."):
        return key
    if key.startswith("bottleneck.vq.codebook."):
        return "vq." + key.removeprefix("bottleneck.vq.codebook.")
    if key.startswith("trunk."):
        return "decoder." + key.removeprefix("trunk.")
    if key.startswith("head.norm."):
        return "decoder.norm_out." + key.removeprefix("head.norm.")
    if key.startswith("head.proj."):
        target = "conv_out." if conv_trunk else "out_proj."
        return "decoder." + target + key.removeprefix("head.proj.")
    raise KeyError(f"no legacy form for {key}")


def to_legacy(
    state: dict[str, torch.Tensor], conv_trunk: bool
) -> dict[str, torch.Tensor]:
    """New-layout state to plain legacy keys."""
    return {legacy_key(k, conv_trunk): v for k, v in state.items()}


def split_lightning(state: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """A Lightning `state_dict` to `(live, ema)`, both plain legacy state."""
    live, ema = {}, {}
    for key, value in state.items():
        if key.startswith(EMA_PREFIX):
            index, _, rest = key.removeprefix(EMA_PREFIX).partition("__")
            ema[EMA_TARGETS[index] + rest.replace("__", ".")] = value
        elif key.startswith("vq_handler.vq."):
            live["vq." + key.removeprefix("vq_handler.vq.")] = value
        elif key.startswith(("encoder.", "decoder.")):
            live[key] = value
        elif not key.startswith(LIGHTNING_DROPPED) and key != "ema.num_updates":
            raise KeyError(f"unknown Lightning key {key}")
    return live, ema
