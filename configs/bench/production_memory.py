"""Production at its batch with the GAN gate open from step 0: runnable with and without
gradient checkpoints? One GPU each, synthetic data."""

from pathlib import Path

OUT = Path("runs/bench/production_memory.jsonl")
STEPS = 60
WARM_STEPS = 0
_GATE_OPEN = {
    "name": "taming",
    "disc_start": 0,
    "disc_weight": 0.5,
    "gen_adv_loss": "hinge",
    "disc_loss": "hinge",
    "adaptive_weight": True,
    "adaptive_clip": (0.1, 10000.0),
}
CASES = [
    {
        "recipe": "recipes_kvq_f16c256_production",
        "gpus": [gpu],
        "data": "synthetic",
        "knobs": {"GAN_SCHEME": _GATE_OPEN, "GRAD_CKPT": ckpt},
    }
    for gpu, ckpt in ((0, False), (1, True))
]
