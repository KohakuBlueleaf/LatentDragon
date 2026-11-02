"""State keys of older runs: `bottleneck.` became `latent.` and `trunk.` became `decoder.`."""

import torch

RENAMES = (("bottleneck.", "latent."), ("trunk.", "decoder."))


def upgrade_key(key: str, prefix: str = "") -> str:
    """One key with its old part name replaced; `prefix` precedes the part name."""
    for old, new in RENAMES:
        if key.startswith(prefix + old):
            return prefix + new + key.removeprefix(prefix + old)
    return key


def upgrade_keys(state: dict[str, torch.Tensor], prefix: str = "") -> dict:
    """Every key of `state` upgraded; current keys pass through."""
    return {upgrade_key(key, prefix): value for key, value in state.items()}
