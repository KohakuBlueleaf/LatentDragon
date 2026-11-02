"""latentmaid_f8c4_stylegan2 recipe, carried over from its legacy config.

kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_latentmaid_f8c4_stylegan2.py
"""

from kohakuengine import use_config
from sensecraft import Term

use_config("../data/data_imagenet.py")
use_config("../precision/precision_fp16.py")

# Run identity, architecture and data loading
NAME = "latentmaid_f8c4_stylegan2"
ARCH = "latentmaid_f8c4"
SIZE = 256
DEVICES = [0, 1, 2, 3]
NUM_WORKERS = 16

# Reconstruction losses, auxiliary weights and the adversarial setup
LOSS_CONFIGS = [
    Term("mse"),
    Term("lpips", 0.5, net="vgg"),
]
AUX_WEIGHTS = {"vq": 1.0, "kl": 0.0}
GAN_SCHEME = {
    "name": "stylegan2",
    "disc_start": 50000,
    "disc_weight": 0.5,
    "gamma": 1.0,
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
EPOCHS = 40
GRAD_CLIP = 1.0
GRAD_CKPT = True
COMPILE = "model"
COMPILE_LOSS = True

# Optimizer, schedule and EMA
LR = 0.0001
ENC_LR_SCALE = 1.0
VQ_LR_SCALE = 1.0
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
SCHEDULER_CONFIG = {"mode": "cosine", "min_value": 0.01, "warmup": 0, "end": -1}
EMA_DECAY = 0.9995

# Validation, checkpoint and sample cadence (steps)
VAL_EVERY = 5000
CKPT_EVERY = 5000
SAMPLE_EVERY = 1000
