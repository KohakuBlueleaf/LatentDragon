"""Vector quantization bottleneck."""

import torch
from torch import Tensor

from latentdragon.bottleneck.base import BOTTLENECK, Bottleneck
from latentdragon.types import BottleneckOut
from latentdragon.vq import QUANTIZER
from latentdragon_common.registry import build


@BOTTLENECK.register("vq")
class VQBottleneck(Bottleneck):
    """Run a channels-last quantizer on a channels-first latent."""

    def __init__(self, latent_dim: int, quantizer: str | dict):
        super().__init__()
        self.vq = build(quantizer, QUANTIZER, emb_dim=latent_dim)

    @property
    def quantizer(self):
        """The grid quantizer."""
        return self.vq

    @property
    def codebook_size(self) -> int:
        """Codes across every partition and level."""
        return self.vq.codebook_size

    def bind_kernels(self, device: torch.device | str) -> None:
        """Forward to the quantizer."""
        self.vq.bind_kernels(device)

    def track_stats(self, on: bool) -> None:
        """Forward to the quantizer."""
        self.vq.track_stats = on

    def forward(self, raw: Tensor, offset: int = 0) -> BottleneckOut:
        """Quantize `raw` and split the quantizer info into loss, metrics and indices."""
        # Channels-first to channels-last for the quantizer
        out, info = self.vq(raw.movedim(1, -1))

        indices = info.pop("indices")
        vq_loss = info.pop("vq_loss")
        return BottleneckOut(
            z=out.movedim(-1, 1),
            aux={"vq": vq_loss},
            metrics={key: value.detach() for key, value in info.items()},
            indices=indices,
            raw=raw,
        )
