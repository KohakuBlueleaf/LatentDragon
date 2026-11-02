"""PatchGAN with smooth activations and per-sample norms, safe for R1 and R2."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from latentdragon.gan.critics.base import CRITIC, Critic
from latentdragon_common.calls import identity


class RMSNorm2d(nn.Module):
    """Channel-wise RMSNorm on NCHW maps."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.square().mean(1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight[None, :, None, None]


def _norm_layer(norm: str | None, channels: int) -> nn.Module:
    match norm:
        case "rms":
            return RMSNorm2d(channels)
        case "group":
            return nn.GroupNorm(min(32, channels), channels)
        case None | "none":
            return nn.Identity()
        case _:
            raise ValueError(f"unknown norm {norm!r}")


@CRITIC.register("modern_patchgan")
class ModernPatchGAN(Critic):
    """Same depth and width as PatchGAN, with no batch coupling."""

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm: str | None = "rms",
        act: str = "silu",
        use_spectral_norm: bool = False,
        slot: str = "x",
    ):
        super().__init__(slot)
        kw, padw = 4, 1
        self.tracks_stats = use_spectral_norm
        act_cls = {"silu": nn.SiLU, "mish": nn.Mish}[act]
        wrap = spectral_norm if use_spectral_norm else identity
        blocks = [nn.Sequential(wrap(nn.Conv2d(input_nc, ndf, kw, 2, padw)), act_cls())]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2**n, 8)
            blocks.append(
                nn.Sequential(
                    wrap(nn.Conv2d(ndf * prev, ndf * mult, kw, 2, padw, bias=False)),
                    _norm_layer(norm, ndf * mult),
                    act_cls(),
                )
            )
        prev, mult = mult, min(2**n_layers, 8)
        blocks.append(
            nn.Sequential(
                wrap(nn.Conv2d(ndf * prev, ndf * mult, kw, 1, padw, bias=False)),
                _norm_layer(norm, ndf * mult),
                act_cls(),
            )
        )
        blocks.append(wrap(nn.Conv2d(ndf * mult, 1, kw, 1, padw)))
        self.main = nn.Sequential(*blocks)

    def judge(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
