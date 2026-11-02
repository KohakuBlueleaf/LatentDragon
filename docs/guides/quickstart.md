# Quick start

1. Install on Python 3.13: `uv pip install -e ".[kernels,wandb]"`.
2. Set the roots in `configs/ae/data/imagenet.py` to your ImageNet `train` and `val` folders.
3. Pick a recipe under `configs/ae/recipes/` and run it:

```bash
kogine run scripts/train/ae.py --config configs/ae/recipes/latentmaid/0222.py
```

Override any knob with `--set NAME=value`. The first GPU run tunes kernels per card and
stores the table in the user cache; set `KERNEL_TUNE=off` to bind the first candidate.

Two defaults worth keeping: use `PRECISION="bf16"`, and raise `CKPT_EVERY` or set `CKPT_KEEP` on
long runs, since a checkpoint holds optimizer state and EMA.
