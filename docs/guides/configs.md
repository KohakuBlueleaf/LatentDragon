# Configs

The script owns every knob: UPPER_CASE globals in `scripts/train/ae.py`, whose defaults are
a CPU smoke recipe. Unknown knobs are refused with a did-you-mean.

Layers live in `configs/ae/`: `data/`, `precision/`, and `recipes/`, which compose them with
`use_config`. Files are named `<group>_<name>.py` so stems stay identifiers. `use_config`
replaces whole values, dicts included.

Dataset roots come from environment variables, never from tracked paths:
`LATENTDRAGON_DATA`, `LATENTDRAGON_TRAIN_ROOTS` (separated by the OS path separator) and
`LATENTDRAGON_STAGE1`.

Architecture is `ARCH`, a name in `PRESETS`, plus `ARCH_OVERRIDES`, field replacements on
`AEArchConfig`.
