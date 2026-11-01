"""`GanScheme`: the policy record a task calls once per micro-batch."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from latentdragon_common.registry import Registry

GAN_SCHEME = Registry("gan scheme")


@dataclass
class GanMicro:
    """One decoded view of one micro-batch."""

    real: torch.Tensor
    fake: torch.Tensor
    recon_seed: torch.Tensor
    aux: list[tuple[float, torch.Tensor]] = field(default_factory=list)


@dataclass
class GanCtx:
    """What a scheme needs from the task for one call."""

    m: GanMicro
    micro_index: int
    micro_count: int
    critic: torch.nn.Module | None
    backward_g: Callable[[torch.Tensor], None]
    backward_d: Callable[[torch.Tensor], None]
    anchor: torch.Tensor
    sync: object
    seed_scale: float = 1.0


def seed_of(
    loss: torch.Tensor,
    wrt: torch.Tensor,
    scale: float = 1.0,
    retain_graph: bool = False,
) -> torch.Tensor:
    """`d loss / d wrt`, taken scaled so low-precision graphs do not underflow."""
    return torch.autograd.grad(loss * scale, wrt, retain_graph=retain_graph)[0] / scale


def aux_scalar(m: GanMicro, micro_count: int) -> torch.Tensor | None:
    """The weighted live scalars of a micro, each divided by `micro_count`."""
    total = None
    for weight, value in m.aux:
        term = (weight / micro_count) * value
        total = term if total is None else total + term
    return total


def composed_scalar(m: GanMicro, seed: torch.Tensor, micro_count: int) -> torch.Tensor:
    """`(fake * seed).sum()` plus the micro's weighted auxiliary scalars."""
    scalar = (m.fake * seed).sum()
    extra = aux_scalar(m, micro_count)
    return scalar if extra is None else scalar + extra


def gen_backward(ctx: GanCtx, adv_term: torch.Tensor | None) -> None:
    """The single composed generator backward for one micro-batch."""
    seed = ctx.m.recon_seed / ctx.micro_count
    if adv_term is not None:
        seed = seed + adv_term
    ctx.backward_g(composed_scalar(ctx.m, seed, ctx.micro_count))


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

    def for_critic(self, critic) -> "GanScheme":
        """This scheme specialised for a built critic; default is unchanged."""
        return self


@GAN_SCHEME.register("none")
def no_gan(**_unused) -> GanScheme:
    """A pure autoencoder: no critic forward, ever."""
    return GanScheme("none", 0, recon_only, recon_only, uses_critic=False)
