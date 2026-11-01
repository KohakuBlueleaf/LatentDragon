"""Kernel ops: `resolve(op, device)` binds one implementation at build time.

Op signatures live in `ops`. Importing this package never imports triton.
"""

from latentdragon_kernels.errors import NoImplementationError
from latentdragon_kernels.registry import register, resolve

__all__ = [
    # errors
    "NoImplementationError",
    # registry functions
    "register",
    "resolve",
]
