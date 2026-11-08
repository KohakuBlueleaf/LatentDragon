"""Config of a LatentDragon autoencoder on the Hugging Face Hub.

Shipped as remote code next to the weights. It needs only `transformers`; LatentDragon itself
reads the same `config.json` through `latentdragon.io.hf.from_pretrained`.
"""

from transformers import PretrainedConfig


class LatentDragonConfig(PretrainedConfig):
    """The LatentDragon `ARCH` dict, plus summary fields for readers that skip it.

    `arch` is the full build spec. The summary fields (`latent_channels`, `spatial_scale`,
    `partitions`, `partition_channels`, `codebook_size`, `rvq_levels`, `nested_partitions`)
    are written at export and never read back by the model.
    """

    model_type = "latentdragon"

    def __init__(self, arch: dict | None = None, **kwargs):
        self.arch = arch or {}
        super().__init__(**kwargs)
