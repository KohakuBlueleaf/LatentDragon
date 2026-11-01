"""`FREEZE`: mark model parts as fixed, keeping them in eval mode."""

import torch


def frozen_parts(model: torch.nn.Module, prefixes: list[str]) -> list[torch.nn.Module]:
    """Resolve dotted submodule paths; an unknown path raises."""
    return [model.get_submodule(prefix) for prefix in prefixes]


def apply_freeze(parts: list[torch.nn.Module]) -> None:
    """Stop gradients and hold evaluation behaviour for every part."""
    for part in parts:
        part.requires_grad_(False)
        part.eval()
