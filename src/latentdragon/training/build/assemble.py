"""Build-time assembly: knobs resolve to a task, bundles and callbacks once."""

import math
from pathlib import Path

import torch
from anyschedule import AnySchedule

from latentdragon.gan import CRITIC, GAN_SCHEME
from latentdragon.latent import LATENT_TERM, LATENT_VIEW, TEACHER, WEIGHTING
from latentdragon.model import build_autoencoder, get_preset
from latentdragon.model.mechanics import apply_compile, apply_recompute
from latentdragon.objectives import HEAD_OBJECTIVES, OBJECTIVE
from latentdragon.tasks import TASK, AutoencoderParts
from latentdragon.tasks.freeze import frozen_parts
from latentdragon.training.bundle import Bundle
from latentdragon.training.checkpoint import latest
from latentdragon.training.checkpoint.partial import (
    load_partial,
    refuse_frozen_missing,
)
from latentdragon.training.checks.problems import spec_name
from latentdragon.training.ema import NamedEMA, NoEMA
from latentdragon.training.parallel import sync
from latentdragon.training.parallel.reducer import MeanReducer, NoReducer
from latentdragon.training.precision import PRECISION
from latentdragon.training.scaler import DynamicScaler, NoScaler
from latentdragon_common.registry import Registry, build
from latentdragon_kernels.tuning import tuner

OPTIMIZER = Registry("optimizer")
OPTIMIZER.add("adam", torch.optim.Adam)
OPTIMIZER.add("adamw", torch.optim.AdamW)
LOSS_MODES = {"image": "2d", "video": "3d"}
WIRE_DTYPES = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}


def arch_config(k: dict):
    """The selected architecture with overrides applied."""
    return get_preset(k["ARCH"], **k["ARCH_OVERRIDES"])


def _terms(k: dict) -> list:
    """`(term, weighting)` pairs for every `LATENT_TERMS` entry."""
    return [
        (build(spec["term"], LATENT_TERM), build(spec.get("weight"), WEIGHTING))
        for spec in k["LATENT_TERMS"]
    ]


def _objective(k: dict, arch):
    """The `OBJECTIVE` knob, or the one paired with the head, in the domain's loss mode."""
    name = k["OBJECTIVE"] or HEAD_OBJECTIVES[spec_name(arch.head)]
    return build(
        name,
        OBJECTIVE,
        loss_configs=k["LOSS_CONFIGS"],
        mode=LOSS_MODES[arch.domain],
    )


def build_task(k: dict, device):
    """Model, critic, objective, scheme, views and terms, on `device`."""
    arch = arch_config(k)
    model = build_autoencoder(arch)

    # Seed weights from INIT_FROM unless a checkpoint of a resumed run supersedes them.
    resuming = k.get("RESUME") and latest(Path(k["RESUME"]) / "checkpoints")
    if k["INIT_FROM"] is not None and not resuming:
        report = load_partial(model, k["INIT_FROM"])
        refuse_frozen_missing(report, k["FREEZE"])

    scheme = build(k["GAN_SCHEME"], GAN_SCHEME)
    critic = None
    if scheme.uses_critic:
        critic = build(k["CRITIC"], CRITIC, input_nc=arch.image_dim)

    parts = AutoencoderParts(
        model=model,
        objective=_objective(k, arch),
        scheme=scheme,
        critic=critic,
        views=[build(spec, LATENT_VIEW) for spec in k["LATENT_VIEWS"]],
        terms=_terms(k),
        teachers={
            name: build(s, TEACHER).freeze() for name, s in k["TEACHERS"].items()
        },
        aux_weights=k["AUX_WEIGHTS"],
        frozen=frozen_parts(model, k["FREEZE"]),
        enc_lr_scale=k["ENC_LR_SCALE"],
        vq_lr_scale=k["VQ_LR_SCALE"],
        disc_lr_scale=k["DISC_LR_SCALE"],
        metrics_every=k["METRICS_EVERY"],
        usage_window=k["USAGE_WINDOW"],
    )
    task = TASK.get("autoencoder")(parts)

    # Memory/speed transforms, then move everything to the device.
    if k["GRAD_CKPT"]:
        apply_recompute(model)
    if k["COMPILE"]:
        apply_compile(model)
    model.to(device)
    if critic is not None:
        critic.to(device)
    task.term_modules.to(device)

    # Bind kernels on the device, or on cpu when KERNELS is not auto.
    tuner.set_mode(k["KERNEL_TUNE"])
    bind_device = torch.device(device) if k["KERNELS"] == "auto" else "cpu"
    model.bind_kernels(bind_device)
    task.bind_device(device)
    return task


def build_bundles(task, k: dict, total_steps: int) -> dict[str, Bundle]:
    """One bundle per task group, each with its own scaler."""
    precision = build(k["PRECISION"], PRECISION)
    if precision.scaled and any(term.group != "model" for term, _ in task.terms):
        raise ValueError("latent terms need the model group under fp16")

    wire = WIRE_DTYPES[k["REDUCE_DTYPE"]]
    schedule_config = dict(k["SCHEDULER_CONFIG"])
    schedule_config["end"] = total_steps

    bundles = {}
    for name, groups in task.groups().items():
        # Per-group learning rates scale the base LR.
        params = [
            {"params": g["params"], "lr": k["LR"] * g["lr_scale"]} for g in groups
        ]
        optimizer = build(
            k["OPTIMIZER"],
            OPTIMIZER,
            params=params,
            lr=k["LR"],
            betas=k["BETAS"],
            weight_decay=k["WEIGHT_DECAY"],
        )
        bundles[name] = Bundle(
            name,
            optimizer,
            AnySchedule(optimizer, config={"lr": schedule_config}),
            DynamicScaler() if precision.scaled else NoScaler(),
            MeanReducer(wire) if sync.active() else NoReducer(),
            k["GRAD_CLIP"],
        )
    return bundles


def build_ema(task, k: dict):
    """Named EMA, or the null policy."""
    if k["EMA_DECAY"] <= 0:
        return NoEMA()
    return NamedEMA(task.ema_named_params(), k["EMA_DECAY"])


def total_steps(k: dict, dataset_size: int, world: int) -> int:
    """`MAX_STEPS`, or the steps of `EPOCHS` full-batch epochs."""
    if k["MAX_STEPS"] > 0:
        return k["MAX_STEPS"]
    per_epoch = math.floor(dataset_size / world / k["BATCH_SIZE"])
    return per_epoch * k["EPOCHS"]
