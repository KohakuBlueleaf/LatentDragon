"""Several image folders from $LATENTDRAGON_TRAIN_ROOTS (separated by os.pathsep)."""

import os

_ROOT = os.environ.get("LATENTDRAGON_DATA", "data")
_ROOTS = os.environ.get("LATENTDRAGON_TRAIN_ROOTS", f"{_ROOT}/train").split(os.pathsep)
TRAIN_SOURCES = [{"name": "image_folder", "root": root} for root in _ROOTS]
VAL_SOURCE = {"name": "image_folder", "root": f"{_ROOT}/imagenet/val"}
