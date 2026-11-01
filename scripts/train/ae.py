"""Train an autoencoder. Every knob is an UPPER_CASE global; defaults are a CPU smoke run.

kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_kvq_f16c256.py
"""

from latentdragon.training.entry import run

# --- run ---
NAME = "smoke"
RUNS_DIR = "runs"
RUN_ID = None
SEED = 20090220
DEVICES = ["cpu"]  # e.g. [0, 1, 2, 3]
PRECISION = "bf16"  # bf16 | fp16 | fp32
WANDB_PROJECT = None
WANDB_RUN_ID = None  # continue this wandb run
DIST_TIMEOUT_HOURS = 4
RESUME = None  # a run folder to continue
RESUME_IGNORE = []  # knobs allowed to differ on resume
INIT_FROM = None  # weights for a new stage: a safetensors file or checkpoint folder
FREEZE = []  # model parts held fixed, e.g. ["encoder", "bottleneck", "trunk"]

# --- data ---
TRAIN_SOURCES = []  # e.g. [{"name": "image_folder", "root": "..."}]
VAL_SOURCE = None
SIZE = 256
NUM_WORKERS = 8
VAL_WORKERS = 4
VAL_BATCH_SIZE = 64

# --- model ---
ARCH = "kvq_f16c256"
ARCH_OVERRIDES = {}
GRAD_CKPT = False
COMPILE = False
KERNELS = "auto"  # auto | eager
KERNEL_TUNE = "on"  # on times candidates on first use; off binds the first one

# --- objective ---
OBJECTIVE = None  # None selects the objective paired with the head
LOSS_CONFIGS = []
AUX_WEIGHTS = {"vq": 1.0, "kl": 0.0}
GAN_SCHEME = "none"
CRITIC = None

# --- latent slots ---
LATENT_VIEWS = []
LATENT_TERMS = []  # entries: {"term": spec, "weight": spec}
TEACHERS = {}
ALLOW_OFF_CODEBOOK = False

# --- step ---
BATCH_SIZE = 32  # per rank; GRAD_ACC splits it, it does not multiply it
GRAD_ACC = 1
MAX_STEPS = 0
EPOCHS = 0
GRAD_CLIP = 1.0
REDUCE_DTYPE = None  # None | bf16 | fp16: wire dtype of the gradient all-reduce

# --- optimizer ---
OPTIMIZER = "adam"
LR = 2e-4
ENC_LR_SCALE = 1.0
VQ_LR_SCALE = 1.0
DISC_LR_SCALE = 1.0
BETAS = (0.5, 0.95)
WEIGHT_DECAY = 0.0
SCHEDULER_CONFIG = {"mode": "cosine", "min_value": 0.01, "warmup": 0, "end": -1}
EMA_DECAY = 0.999  # 0 turns EMA off

# --- cadence ---
LOG_EVERY = 100
METRICS_EVERY = 10
USAGE_WINDOW = 1000  # steps a code counts as used in vq/usage_ema
SAMPLE_EVERY = 1000
VAL_EVERY = 10000
CKPT_EVERY = 10000
CKPT_KEEP = 0  # 0 keeps every checkpoint
TIMING_EVERY = 0  # drain the device on every Nth step for exact phase times


def main():
    """Run training with this module's knobs."""
    return run(globals(), __file__)


if __name__ == "__main__":
    main()
