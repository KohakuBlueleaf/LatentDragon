"""Several image folders from $LATENTDRAGON_TRAIN_ROOTS (separated by os.pathsep).

Validation reads $LATENTDRAGON_VAL_ROOT, by default `<LATENTDRAGON_DATA>/imagenet/val`.
"""

import os

_ROOT = os.environ.get("LATENTDRAGON_DATA", "data")
_ROOTS = os.environ.get("LATENTDRAGON_TRAIN_ROOTS", f"{_ROOT}/train").split(os.pathsep)
_VAL = os.environ.get("LATENTDRAGON_VAL_ROOT", f"{_ROOT}/imagenet/val")
TRAIN_SOURCES = [{"name": "image_folder", "root": root} for root in _ROOTS]
VAL_SOURCE = {"name": "image_folder", "root": _VAL}
