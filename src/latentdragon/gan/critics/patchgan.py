"""BatchNorm PatchGAN, math-equivalent to the FVQ and taming discriminator."""

import torch
import torch.nn as nn

from latentdragon.gan.critics.base import CRITIC, Critic


def weights_init(module: nn.Module) -> None:
    """Taming init: N(0, 0.02) on convs, N(1, 0.02) and zero bias on BatchNorm."""
    match module:
        case nn.Conv2d():
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        case nn.BatchNorm2d():
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


@CRITIC.register("patchgan")
class PatchGAN(Critic):
    """Pix2Pix PatchGAN; returns the full patch-logit map."""

    tracks_stats = True

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw, padw = 4, 1
        layers = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * prev, ndf * mult, kw, 2, padw, bias=False),
                nn.BatchNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
        prev, mult = mult, min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * prev, ndf * mult, kw, 1, padw, bias=False),
            nn.BatchNorm2d(ndf * mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * mult, 1, kw, 1, padw),
        ]
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
