"""Mixed-precision policies, selected once from the PRECISION knob."""

import contextlib
from dataclasses import dataclass

import torch

from latentdragon_common.registry import Registry

PRECISION = Registry("precision")


@dataclass(frozen=True)
class Precision:
    """Autocast dtype plus whether gradients need dynamic scaling."""

    name: str
    dtype: torch.dtype | None
    scaled: bool

    def autocast(self, device_type: str):
        """The context a micro-step and an eval forward run under."""
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type, dtype=self.dtype)


PRECISION.add("bf16", lambda: Precision("bf16", torch.bfloat16, False))
PRECISION.add("fp16", lambda: Precision("fp16", torch.float16, True))
PRECISION.add("fp32", lambda: Precision("fp32", None, False))
