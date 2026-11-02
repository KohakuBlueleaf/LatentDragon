"""ImageNet layout under $LATENTDRAGON_DATA: `imagenet/train` and `imagenet/val`."""

import os

_ROOT = os.environ.get("LATENTDRAGON_DATA", "data")
TRAIN_SOURCES = [{"name": "image_folder", "root": f"{_ROOT}/imagenet/train"}]
VAL_SOURCE = {"name": "image_folder", "root": f"{_ROOT}/imagenet/val"}
