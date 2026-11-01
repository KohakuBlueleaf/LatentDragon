"""Eager MLP projection: w2 @ act(w1 @ base + b1) + b2."""

import torch
import torch.nn.functional as F
from torch import Tensor

from latentdragon_kernels.backends.eager.quantize import wide
from latentdragon_kernels.ops import MLP_PROJECT
from latentdragon_kernels.registry import register

_ACT = {"silu": F.silu, "gelu": F.gelu, "mish": F.mish}


def mlp_project(
    base: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    act: str = "silu",
    trainable_base: bool = True,
) -> Tensor:
    """MLP projection through autograd (see `ops`)."""
    if act not in _ACT:
        raise ValueError(f"act must be one of {list(_ACT)}, got {act!r}")
    dtype = wide(torch.promote_types(base.dtype, w1.dtype))
    source = base if trainable_base else base.detach()
    with torch.autocast(base.device.type, enabled=False):
        hidden = torch.bmm(source.to(dtype), w1.to(dtype)) + b1.to(dtype)
        return torch.bmm(_ACT[act](hidden), w2.to(dtype)) + b2.to(dtype)


register(MLP_PROJECT, "eager", mlp_project)
