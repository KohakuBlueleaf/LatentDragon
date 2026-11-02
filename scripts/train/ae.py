"""Train an autoencoder. Every knob is an UPPER_CASE global; defaults are a CPU smoke run.

kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_kvq_f16c256.py
"""

from typing import Literal

from latentdragon.training.entry import run

# --- run ---
NAME = "smoke"
RUNS_DIR = "runs"
RUN_ID: str | None = None
SEED = 20090220
DEVICES: list[int | str] = ["cpu"]  # e.g. [0, 1, 2, 3]
PRECISION = "bf16"  # bf16 | fp16 | fp32 | bf16-true | fp16-true
CPU_THREADS = 2  # intra-op threads per rank; data workers are separate
WANDB_PROJECT: str | None = None
WANDB_RUN_ID: str | None = None  # continue this wandb run
DIST_TIMEOUT_HOURS = 4
RESUME: str | None = None  # a run folder to continue
RESUME_IGNORE: list[str] = []  # knobs allowed to differ on resume
INIT_FROM: str | None = (
    None  # weights for a new stage: a safetensors file or checkpoint folder
)
FREEZE: list[str] = (
    []
)  # model parts held fixed, e.g. ["encoder", "bottleneck", "trunk"]

# --- data ---
TRAIN_SOURCES: list[dict] = []  # e.g. [{"name": "image_folder", "root": "..."}]
VAL_SOURCE: dict | None = None
SIZE = 256
NUM_WORKERS = 8
VAL_WORKERS = 4
VAL_BATCH_SIZE = 64

# --- model ---
ARCH = "kvq_f16c256"
ARCH_OVERRIDES = {}
GRAD_CKPT = False
COMPILE: Literal["off", "function", "module", "model"] = "off"
COMPILE_MODE = "default"  # torch.compile mode
COMPILE_DYNAMIC = False
COMPILE_TUNE = False  # Inductor coordinate descent tuning
COMPILE_LOSS = False  # also compile the loss networks and the critic
CHANNELS_LAST = False  # image tensors and conv weights in channels-last layout
CUDNN_BENCHMARK = False  # cuDNN times its conv algorithms per shape
KERNELS = "auto"  # auto | eager
KERNEL_TUNE = "on"  # on times candidates on first use; off binds the first one

# --- objective ---
OBJECTIVE: str | None = None  # None selects the objective paired with the head
LOSS_CONFIGS = []
LOSS_16BIT = True  # hold the frozen loss networks in the compute dtype
AUX_WEIGHTS = {"vq": 1.0, "kl": 0.0}
GAN_SCHEME = "none"
CRITIC: dict | None = None

# --- latent slots ---
LATENT_VIEWS: list = []
LATENT_TERMS: list = []  # entries: {"term": spec, "weight": spec}
TEACHERS = {}
ALLOW_OFF_CODEBOOK = False

# --- step ---
BATCH_SIZE = 32  # per rank; GRAD_ACC splits it, it does not multiply it
GRAD_ACC = 1
MAX_STEPS = 0
EPOCHS = 0
GRAD_CLIP = 1.0
REDUCE_DTYPE: str | None = (
    None  # None | bf16 | fp16: wire dtype of the gradient all-reduce
)
REDUCER = {"name": "bucketed", "bucket_mib": 8.0}  # bucketed | flat | none

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
