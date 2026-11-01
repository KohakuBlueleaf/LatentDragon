"""Convert a legacy checkpoint to a LatentDragon export folder.

kogine run scripts/tools/convert_legacy.py --set SOURCE=... --set TARGET=...
"""

from pathlib import Path

from latentdragon.io.export import save_export
from latentdragon.io.legacy import read_lightning, read_safetensors

SOURCE = ""  # a Lightning .ckpt or a legacy .safetensors
TARGET = "converted"
USE_EMA = True


def main():
    """Read the legacy checkpoint (EMA laid over live weights) and save an export."""
    if Path(SOURCE).suffix == ".safetensors":
        arch, state = read_safetensors(SOURCE)
    else:
        arch, live, ema = read_lightning(SOURCE)
        state = {**live, **ema} if USE_EMA else live
    save_export(TARGET, arch, state)


if __name__ == "__main__":
    main()
