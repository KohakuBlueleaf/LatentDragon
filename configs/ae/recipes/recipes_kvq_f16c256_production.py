"""kvq_f16c256_production recipe, carried over from its legacy config.


kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_kvq_f16c256_production.py
"""

from kohakuengine import use_config
from sensecraft import Term

use_config("../data/data_multi.py")
use_config("../precision/precision_fp16.py")

# Run identity, architecture and data loading
NAME = "kvq_f16c256_production"
ARCH = "kvq_f16c256_production"
SIZE = 256
DEVICES = [0, 1, 2, 3]
NUM_WORKERS = 16

# Reconstruction losses, auxiliary weights and the adversarial setup
LOSS_CONFIGS = [
    Term("mse"),
    Term("lpips", 0.5, net="vgg"),
    Term("dino_vit", 250, model="base", use_gram=False),
]
AUX_WEIGHTS = {"vq": 1.0, "kl": 0.0}
GAN_SCHEME = {
    "name": "taming",
    "disc_start": 50000,
    "disc_weight": 0.5,
    "gen_adv_loss": "hinge",
    "disc_loss": "hinge",
    "adaptive_weight": True,
    "adaptive_clip": (0.1, 10000.0),
}
CRITIC = {
    "name": "modern_patchgan",
    "ndf": 64,
    "n_layers": 3,
    "norm": "rms",
    "use_spectral_norm": False,
}

# Loop length and memory
BATCH_SIZE = 32
GRAD_ACC = 1
EPOCHS = 1
GRAD_CLIP = 1.0
GRAD_CKPT = False
COMPILE = "model"
COMPILE_LOSS = True

# Optimizer, schedule and EMA
LR = 0.0002
ENC_LR_SCALE = 1.0
VQ_LR_SCALE = 1.0
BETAS = (0.5, 0.95)
WEIGHT_DECAY = 0.0
SCHEDULER_CONFIG = {"mode": "cosine", "min_value": 0.01, "warmup": 0.0, "end": -1}
EMA_DECAY = 0.999

# Validation, checkpoint and sample cadence (steps)
VAL_EVERY = 10000
CKPT_EVERY = 10000
SAMPLE_EVERY = 1000
