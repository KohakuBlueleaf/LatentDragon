"""Op registry: backends register implementations, `resolve` returns one plain callable.

Backends load on the first `resolve`; a backend whose import fails is skipped.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from latentdragon_common.log import get_logger
from latentdragon_kernels.errors import NoImplementationError
from latentdragon_kernels.hardware.arch import arch_of

BACKENDS = ("gemm", "triton", "eager")
log = get_logger("latentdragon.kernels")


@dataclass
class Impl:
    """One backend's implementation of one op."""

    fn: Callable
    supports: Callable[[str], str | None] | None = None


@dataclass
class State:
    """Everything registered in this process; an op holds one list per backend."""

    impls: dict[str, dict[str, list[Impl]]] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    loaded: bool = False


STATE = State()


def register(
    op: str,
    backend: str,
    fn: Callable,
    supports: Callable[[str], str | None] | None = None,
) -> None:
    """Add `fn` to `backend`'s implementations of `op`; newest first in `resolve`."""
    STATE.impls.setdefault(op, {}).setdefault(backend, []).insert(0, Impl(fn, supports))


def mark_unavailable(backend: str, reason: str) -> None:
    """Record why a whole backend cannot be used in this process."""
    STATE.unavailable[backend] = reason


def load_backends() -> None:
    """Import every backend package once; an ImportError marks that backend unavailable."""
    if STATE.loaded:
        return
    STATE.loaded = True
    for name in BACKENDS:
        try:
            importlib.import_module(f"latentdragon_kernels.backends.{name}")
        except ImportError as err:
            mark_unavailable(name, str(err))


def _pick(op: str, backend: str, arch: str) -> tuple[Impl | None, str]:
    """The first of `backend`'s `op` implementations serving `arch`, or why there is none."""
    if backend in STATE.unavailable:
        return None, f"unavailable ({STATE.unavailable[backend]})"
    reasons = []
    for impl in STATE.impls.get(op, {}).get(backend, ()):
        reason = None if impl.supports is None else impl.supports(arch)
        if reason is None:
            return impl, ""
        reasons.append(reason)
    return None, "; ".join(reasons) or "not implemented"


def resolve(
    op: str, device: torch.device | str, backend: str | None = None
) -> Callable:
    """The implementation of `op` on `device`; a pinned `backend` is the only option."""
    load_backends()
    arch = arch_of(device)
    reasons = []

    # First backend in priority order that can serve the op on this arch
    for name in (backend,) if backend else BACKENDS:
        impl, why = _pick(op, name, arch)
        if impl is not None:
            log.info("kernel bound", op=op, arch=arch, backend=name)
            return impl.fn
        reasons.append(f"{name}: {why}")
    raise NoImplementationError(f"{op} on {arch}: " + "; ".join(reasons))
