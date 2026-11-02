"""The video preset on 13-frame clips, with a mixed reconstruction loss."""

from kohakuengine import use_config
from sensecraft.loss import GeneralConfig, LPIPSConfig

use_config("../precision/precision_bf16.py")

NAME = "bench_video"
ARCH = "video_f16t6c32"
SIZE = 256
FRAMES = 13
BATCH_SIZE = 4
GRAD_CKPT = True
LOSS_CONFIGS = [GeneralConfig("mse", weight=1.0), LPIPSConfig(net="vgg", weight=0.5)]
AUX_WEIGHTS = {"vq": 1.0, "kl": 0.0}
