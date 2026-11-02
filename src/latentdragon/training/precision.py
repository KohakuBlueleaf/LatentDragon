"""Precision policies, selected once from the PRECISION knob.

`bf16` and `fp16` keep fp32 weights under autocast. `bf16-true` and `fp16-true` hold the
weights in 16 bits too, which removes the weight casts and halves weights and gradients.
"""

import contextlib
from dataclasses import dataclass

import torch

from latentdragon_common.registry import Registry

PRECISION = Registry("precision")


@dataclass(frozen=True)
class Precision:
    """Compute dtype, whether gradients need scaling, and the dtype of the weights."""

    name: str
    dtype: torch.dtype | None
    scaled: bool
    weights: torch.dtype | None = None

    def autocast(self, device_type: str):
        """The context a micro-step and an eval forward run under."""
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type, dtype=self.dtype)


PRECISION.add("bf16", lambda: Precision("bf16", torch.bfloat16, False))
PRECISION.add("fp16", lambda: Precision("fp16", torch.float16, True))
PRECISION.add("fp32", lambda: Precision("fp32", None, False))
PRECISION.add(
    "bf16-true", lambda: Precision("bf16-true", torch.bfloat16, False, torch.bfloat16)
)
PRECISION.add(
    "fp16-true", lambda: Precision("fp16-true", torch.float16, True, torch.float16)
)


def cast_weights(module: torch.nn.Module, dtype: torch.dtype, keep: tuple) -> dict:
    """Move parameters to `dtype` in place, except names holding an entry of `keep` and the
    parameters of modules that set `fp32 = True`.

    Returns how many values moved and stayed, so an empty `keep` match shows in the log.
    """
    held = {
        id(p)
        for m in module.modules()
        if getattr(m, "fp32", False) is True
        for p in m.parameters()
    }
    moved, kept = 0, 0
    for name, param in module.named_parameters():
        if id(param) in held or any(token in name for token in keep):
            kept += param.numel()
            continue
        param.data = param.data.to(dtype)
        moved += param.numel()
    return {"cast": moved, "kept_fp32": kept}
