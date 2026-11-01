"""The `image` domain: `(B, C, H, W)`."""

from collections.abc import Callable
from functools import partial

import torch.nn.functional as F
import torchvision.transforms as trns
from torch import Tensor

from latentdragon.domain.record import DOMAIN, Domain, sample_grid, save_image_grid
from latentdragon_common.calls import identity


def image_transforms(size: int, train: bool) -> Callable:
    """PIL image to a normalized tensor; random crop for training."""
    crop = trns.RandomCrop(size) if train else trns.CenterCrop(size)
    return trns.Compose(
        [
            trns.Resize(size),
            crop,
            trns.ToTensor(),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def fit_image_latents(latents: Tensor, real: Tensor) -> Tensor:
    """Scale rendered latents to the sample size."""
    return F.interpolate(latents, size=real.shape[-2:], mode="nearest")


IMAGE = Domain(
    name="image",
    rank=4,
    transforms=image_transforms,
    metrics=("psnr", "ssim", "rfid"),
    to_grid=partial(sample_grid, fit_image_latents),
    to_frames=identity,
    save_grid=save_image_grid,
)


@DOMAIN.register("image")
def image_domain() -> Domain:
    """The image record."""
    return IMAGE
