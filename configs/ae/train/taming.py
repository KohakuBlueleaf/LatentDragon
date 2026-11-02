"""The taming training setup: MSE + LPIPS, KL, hinge PatchGAN with adaptive weight."""

from sensecraft import Term

# Run identity and data loading
WANDB_PROJECT = "LatentDragon-img"
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
    "disc_start": 25000,
    "disc_weight": 0.5,
    "gen_adv_loss": "hinge",
    "disc_loss": "hinge",
    "adaptive_weight": True,
    "adaptive_clip": (0.1, 10000.0),
}
CRITIC = {"name": "patchgan", "ndf": 64, "n_layers": 3}

# Loop length and memory
BATCH_SIZE = 16
GRAD_ACC = 1
EPOCHS = 13
GRAD_CLIP = 1.0
GRAD_CKPT = True
COMPILE = "model"
COMPILE_LOSS = True
CHANNELS_LAST = True
CUDNN_BENCHMARK = True

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
