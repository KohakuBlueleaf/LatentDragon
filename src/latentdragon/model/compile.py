"""`torch.compile` scope, chosen by the `COMPILE` knob.

| scope | compiled |
| --- | --- |
| `off` | nothing |
| `function` | each layer forward: norms, SwiGLU, attention |
| `module` | each block forward, its layers inlined |
| `model` | encoder, trunk and head whole, every block inlined |

The scopes never conflict: a decorated forward reached inside a compiled graph runs as plain
Python and inlines. The bottleneck stays out, because its kernels launch Triton.
"""

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch._functorch.config
import torch._inductor.config
from torch import nn

SCOPES = ("off", "function", "module", "model")
ACTIVE = {"function": ("function", "module"), "module": ("module",)}


@dataclass
class CompileSetup:
    """The selected scope and `torch.compile` arguments; `generation` drops old graphs."""

    scope: str = "off"
    mode: str = "default"
    dynamic: bool = False
    generation: int = 0

    def kwargs(self) -> dict:
        """Keyword arguments of every `torch.compile` call."""
        return {"mode": self.mode, "dynamic": self.dynamic}


SETUP = CompileSetup()


def configure(
    scope: str = "off",
    mode: str = "default",
    dynamic: bool = False,
    tune: bool = False,
    fallback_random: bool = False,
) -> None:
    """Select the scope for every wrapper; takes effect at each wrapper's next call."""
    if scope not in SCOPES:
        raise ValueError(f"compile scope {scope!r} not in {SCOPES}")
    SETUP.scope = scope
    SETUP.mode = mode
    SETUP.dynamic = dynamic
    SETUP.generation += 1
    torch._inductor.config.coordinate_descent_tuning = tune
    torch._inductor.config.fallback_random = fallback_random


def allow_donated_buffers(allowed: bool) -> None:
    """Whether compiled backwards may reuse their inputs; off for a retained graph."""
    torch._functorch.config.donated_buffer = allowed


def compile_wrapper(func: Callable | None = None, *, scope: str = "function"):
    """`func` compiled once, when the selected scope covers `scope`.

    Usable as `@compile_wrapper` or `@compile_wrapper(scope="module")`. A `function` forward
    also compiles under `module`, where it inlines into the enclosing block's graph.
    """
    if func is None:
        return functools.partial(compile_wrapper, scope=scope)

    active = ACTIVE[scope]
    run, generation = func, -1

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal run, generation

        # Traced into an enclosing graph: inline, with no lazy assignment
        if torch.compiler.is_compiling():
            return func(*args, **kwargs)

        if generation != SETUP.generation:
            compiled = SETUP.scope in active
            run = torch.compile(func, **SETUP.kwargs()) if compiled else func
            generation = SETUP.generation
        return run(*args, **kwargs)

    return wrapper


def compile_parts(parts: Iterable[nn.Module]) -> None:
    """Under `model` scope, compile each module whole, in place."""
    if SETUP.scope != "model":
        return
    for part in parts:
        part.compile(**SETUP.kwargs())
