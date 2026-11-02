"""Arch tags: `smXX` for a CUDA device, `cpu` for everything else, and their families.

A family is the unit a kernel targets: cards in one share an instruction set and a tensor
core, so a kernel or a tuned config moves between them. `consumer` and `datacenter` Blackwell
are separate because only the datacenter parts have tensor memory.
"""

import torch

FAMILIES = {
    70: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    87: "ampere",
    89: "ada",
    90: "hopper",
    100: "blackwell-datacenter",
    103: "blackwell-datacenter",
    120: "blackwell-consumer",
}


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


def family_of(arch: str) -> str:
    """The kernel family of an arch tag; an unknown card is its own family."""
    if arch == "cpu":
        return "cpu"
    return FAMILIES.get(sm_of(arch), arch)


def sm_of(arch: str) -> int:
    """The shader model number of an arch tag, 0 for `cpu`."""
    return 0 if arch == "cpu" else int(arch.removeprefix("sm"))
