# Configs

The script owns every knob: UPPER_CASE globals in `scripts/train/ae.py`, whose defaults are
a CPU smoke recipe. A config is Python, so it can define its own helper variables; an UPPER_CASE
name the script does not read is logged as a warning with a did-you-mean, and the run goes on.

Layers live in `configs/ae/`: `arch/`, `data/`, `precision/`, `train/`, and `recipes/`, which
compose them with `use_config`. The folder names the group, so files are named for what they hold
(`precision/fp16.py`, `recipes/latentmaid/0222.py`). `use_config`
replaces whole values, dicts included.

Data is plain config: `TRAIN_SOURCES` is a list of sources, one per folder, and `VAL_SOURCE`
is one source. `configs/ae/data/imagenet.py` names `data/imagenet/train` and `data/imagenet/val`;
change the roots there or set both knobs in your own config.

Architecture is `ARCH`: the parts `domain`, `encoder`, `latent`, `decoder` and `head`, each
a registry choice with its own keyword arguments (see [model](../concepts/model.md)).
`configs/ae/arch/` holds the published layouts (`sdvae.py`, `fluxvae.py`,
`latentmaid/{2222,0222,0022}.py`); a recipe pulls one in with `use_config` or writes its own
`ARCH`.

`LOSS_CONFIGS` is a list of sensecraft terms for a single-slot domain, or a dict of them per data
slot. `AUX_WEIGHTS` weighs the latent def's losses by name. `LR_SCALES` scales the learning rate
of the `encoder`, `latent`, `decoder`, `head` and `critic` groups.
