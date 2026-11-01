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

BACKENDS = ("triton", "eager")
log = get_logger("latentdragon.kernels")


@dataclass
class Impl:
    """One backend's implementation of one op."""

    fn: Callable
    supports: Callable[[str], str | None] | None = None


@dataclass
class State:
    """Everything registered in this process."""

    impls: dict[str, dict[str, Impl]] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    loaded: bool = False


STATE = State()


def register(
    op: str,
    backend: str,
    fn: Callable,
    supports: Callable[[str], str | None] | None = None,
) -> None:
    """Register `fn` as `backend`'s implementation of `op`."""
    STATE.impls.setdefault(op, {})[backend] = Impl(fn, supports)


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


def _reason(op: str, backend: str, arch: str) -> str | None:
    """Why `backend` cannot serve `op` on `arch`, or None."""
    if backend in STATE.unavailable:
        return f"unavailable ({STATE.unavailable[backend]})"
    impl = STATE.impls.get(op, {}).get(backend)
    if impl is None:
        return "not implemented"
    return None if impl.supports is None else impl.supports(arch)


def resolve(
    op: str, device: torch.device | str, backend: str | None = None
) -> Callable:
    """The implementation of `op` on `device`; a pinned `backend` is the only option."""
    load_backends()
    arch = arch_of(device)
    reasons = []

    # First backend in priority order that can serve the op on this arch
    for name in (backend,) if backend else BACKENDS:
        why = _reason(op, name, arch)
        if why is None:
            log.info("kernel bound", op=op, arch=arch, backend=name)
            return STATE.impls[op][name].fn
        reasons.append(f"{name}: {why}")
    raise NoImplementationError(f"{op} on {arch}: " + "; ".join(reasons))
