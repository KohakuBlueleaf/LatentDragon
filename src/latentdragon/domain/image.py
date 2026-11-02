"""The `image` domain: one slot of `(B, C, H, W)` in [-1, 1]."""

from collections.abc import Callable
from functools import partial

import torch.nn.functional as F
import torchvision.transforms as trns
from torch import Tensor

from latentdragon.domain.record import (
    DOMAIN,
    Domain,
    ToSlot,
    sample_grid,
    save_image_grid,
)
from latentdragon_common.calls import identity


def image_transforms(size: int, train: bool, slot: str = "x") -> Callable:
    """PIL image to `{slot: tensor}`; random crop for training."""
    crop = trns.RandomCrop(size) if train else trns.CenterCrop(size)
    pipeline = trns.Compose(
        [
            trns.Resize(size),
            crop,
            trns.ToTensor(),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return ToSlot(slot, pipeline)


def fit_image_latents(latents: Tensor, real: Tensor) -> Tensor:
    """Scale rendered latents to the sample size."""
    return F.interpolate(latents, size=real.shape[-2:], mode="nearest")


@DOMAIN.register("image")
def image_domain(slot: str = "x") -> Domain:
    """Images under one slot, `x` by default."""
    return Domain(
        name="image",
        slots=(slot,),
        transforms=partial(image_transforms, slot=slot),
        loss_dims={slot: 2},
        metrics={slot: ("psnr", "ssim", "rfid")},
        to_frames={slot: identity},
        to_grid=partial(sample_grid, fit_image_latents, slot),
        save_grid=save_image_grid,
    )
