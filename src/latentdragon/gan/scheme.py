"""`GanScheme`: the policy record a task calls once per micro-batch."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from latentdragon.model.core import tree
from latentdragon_common.registry import Registry

GAN_SCHEME = Registry("gan scheme")


@dataclass
class GanMicro:
    """One decoded view of one micro-batch; `real`, `fake` and `recon_seed` are slot dicts."""

    real: Any
    fake: Any
    recon_seed: Any
    aux: list[tuple[float, torch.Tensor]] = field(default_factory=list)


@dataclass
class GanCtx:
    """What a scheme needs from the task for one call."""

    m: GanMicro
    micro_index: int
    micro_count: int
    critic: torch.nn.Module | None
    backward_g: Callable[..., None]
    backward_d: Callable[..., None]
    anchor: torch.Tensor
    sync: object
    seed_scale: float = 1.0


def seed_of(
    loss: torch.Tensor,
    wrt: Any,
    scale: float = 1.0,
    retain_graph: bool = False,
) -> Any:
    """`d loss / d wrt` over a slot dict, taken scaled so low-precision graphs do not underflow."""
    found = tree.grad(loss * scale, wrt, retain_graph=retain_graph)
    return tree.map_tensors(lambda g: g / scale, found)


def detached(slots):
    """A slot dict with every tensor detached."""
    return tree.map_tensors(lambda t: t.detach(), slots)


def leaf(slots):
    """A slot dict of detached tensors that take gradients."""
    return tree.map_tensors(lambda t: t.detach().requires_grad_(True), slots)


def scaled(seed: Any, factor) -> Any:
    """Every tensor of a seed times `factor`."""
    return tree.map_tensors(lambda g: g * factor, seed)


def added(left: Any, right: Any) -> Any:
    """Two seeds of the same slots, summed tensor by tensor."""
    return tree.like(
        left, [a + b for a, b in zip(tree.leaves(left), tree.leaves(right))]
    )


def aux_scalar(m: GanMicro, micro_count: int) -> torch.Tensor | None:
    """The weighted live scalars of a micro, each divided by `micro_count`."""
    total = None
    for weight, value in m.aux:
        term = (weight / micro_count) * value
        total = term if total is None else total + term
    return total


def composed_scalar(m: GanMicro, seed: torch.Tensor, micro_count: int) -> torch.Tensor:
    """`(fake * seed).sum()` over every slot plus the micro's weighted auxiliary scalars."""
    scalar = tree.dot(m.fake, seed)
    extra = aux_scalar(m, micro_count)
    return scalar if extra is None else scalar + extra


def gen_backward(ctx: GanCtx, adv_term: torch.Tensor | None) -> None:
    """The single composed generator backward for one micro-batch."""
    seed = scaled(ctx.m.recon_seed, 1 / ctx.micro_count)
    if adv_term is not None:
        seed = added(seed, adv_term)
    final = ctx.micro_index == ctx.micro_count - 1
    ctx.backward_g(composed_scalar(ctx.m, seed, ctx.micro_count), final=final)


def recon_only(ctx: GanCtx) -> dict:
    """The generator backward with no adversarial term and no critic forward."""
    gen_backward(ctx, None)
    return {}


def no_problems(knobs: dict) -> list[str]:
    """A scheme with nothing to validate."""
    return []


@dataclass(frozen=True)
class GanScheme:
    """Selected adversarial policy: `active` from `start` on, `idle` before it."""

    name: str
    start: int
    active: Callable[[GanCtx], dict]
    idle: Callable[[GanCtx], dict] = recon_only
    uses_critic: bool = True
    problems: Callable[[dict], list[str]] = no_problems
    second_order: bool = (
        False  # backward through a gradient, which no compiled graph does
    )
    retained_backward: bool = False  # walks one graph twice under `retain_graph`
    needs_anchor: bool = False  # reads gradients at the head's `last_layer` weight

    def for_critic(self, critic) -> "GanScheme":
        """This scheme specialised for a built critic; default is unchanged."""
        return self


@GAN_SCHEME.register("none")
def no_gan(**_unused) -> GanScheme:
    """A pure autoencoder: no critic forward, ever."""
    return GanScheme("none", 0, recon_only, recon_only, uses_critic=False)
