"""latentmaid_0222_f16c64n16 recipe, carried over from its legacy config.

kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_latentmaid_0222_f16c64n16.py
"""

from kohakuengine import use_config
from sensecraft import Term

use_config("../data/data_imagenet.py")
use_config("../precision/precision_fp16.py")

# Run identity, architecture and data loading
NAME = "latentmaid_0222_f16c64n16"
ARCH = "latentmaid_0222_f16c64n16"
SIZE = 256
DEVICES = [0, 1, 2, 3]
NUM_WORKERS = 16

# Reconstruction losses, auxiliary weights and the adversarial setup
LOSS_CONFIGS = [
    Term("mse"),
    Term("lpips", 0.5, net="vgg"),
]
AUX_WEIGHTS = {"vq": 1.0, "kl": 1e-06}
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
BATCH_SIZE = 16
GRAD_ACC = 1
EPOCHS = 20
GRAD_CLIP = 1.0
GRAD_CKPT = True
COMPILE = "model"
COMPILE_LOSS = True

# Optimizer, schedule and EMA
LR = 0.000192
ENC_LR_SCALE = 1.0
VQ_LR_SCALE = 1.0
BETAS = (0.5, 0.9)
WEIGHT_DECAY = 0.0
SCHEDULER_CONFIG = {"mode": "constant", "min_value": 0.01, "warmup": 0, "end": -1}
EMA_DECAY = 0.9995

# Validation, checkpoint and sample cadence (steps)
VAL_EVERY = 2500
CKPT_EVERY = 5000
SAMPLE_EVERY = 1000
