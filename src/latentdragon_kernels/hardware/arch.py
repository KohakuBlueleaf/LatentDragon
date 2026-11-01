"""Arch tags: `smXX` for a CUDA device, `cpu` for everything else."""

import torch


def arch_of(device: torch.device | str) -> str:
    """The arch tag of `device`."""
    device = torch.device(device)
    if device.type != "cuda":
        return "cpu"
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major * 10 + minor}"


def kind_of(arch: str) -> str:
    """The device kind of an arch tag."""
    return "cpu" if arch == "cpu" else "cuda"


def sm_of(arch: str) -> int:
    """The shader model number of an arch tag, 0 for `cpu`."""
    return 0 if arch == "cpu" else int(arch.removeprefix("sm"))
