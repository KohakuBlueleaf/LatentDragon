"""Build-time assembly: knobs resolve to a task, bundles and callbacks once."""

import functools
import math
from pathlib import Path

import torch
from anyschedule import AnySchedule

from latentdragon.gan import CRITIC, GAN_SCHEME
from latentdragon.latent import LATENT_TERM, LATENT_VIEW, TEACHER, WEIGHTING
from latentdragon.model import (
    allow_donated_buffers,
    apply_recompute,
    build_autoencoder,
    compile_parts,
    configure,
)
from latentdragon.model.runtime.compile import SETUP
from latentdragon.objectives import OBJECTIVE
from latentdragon.tasks import TASK, AutoencoderParts
from latentdragon.tasks.freeze import frozen_parts
from latentdragon.training.bundle import Bundle
from latentdragon.training.checkpoint import latest
from latentdragon.training.checkpoint.partial import (
    load_partial,
    refuse_frozen_missing,
)
from latentdragon.training.ema import NamedEMA, NoEMA
from latentdragon.training.optim import Adam16
from latentdragon.training.parallel.reducer import REDUCER
from latentdragon.training.precision import PRECISION, cast_weights
from latentdragon.training.scaler import DynamicScaler, NoScaler
from latentdragon_common.log import get_logger
from latentdragon_common.registry import Registry, build
from latentdragon_kernels.tuning import tuner

OPTIMIZER = Registry("optimizer")
OPTIMIZER.add("adam", functools.partial(torch.optim.Adam, fused=True))
OPTIMIZER.add("adamw", functools.partial(torch.optim.AdamW, fused=True))
OPTIMIZER.add("adam16", Adam16)
KEEP_FP32 = ("norm", "bias", "latent")
log = get_logger(__name__)
WIRE_DTYPES = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _terms(k: dict) -> list:
    """`(term, weighting)` pairs for every `LATENT_TERMS` entry."""
    return [
        (build(spec["term"], LATENT_TERM), build(spec.get("weight"), WEIGHTING))
        for spec in k["LATENT_TERMS"]
    ]


def _objective(k: dict, model):
    """The `OBJECTIVE` knob, or the one the head pairs with, over the domain's slots."""
    name = k["OBJECTIVE"] or model.head.objective
    return build(
        name,
        OBJECTIVE,
        loss_configs=k["LOSS_CONFIGS"],
        loss_dims=model.domain.loss_dims,
    )


def _cast_to_policy(k: dict, model, critic) -> None:
    """Hold the weights in 16 bits under a `-true` precision."""
    weights = build(k["PRECISION"], PRECISION).weights
    if weights is None:
        return
    log.info("weights cast", **cast_weights(model, weights, KEEP_FP32))
    if critic is not None:
        cast_weights(critic, weights, KEEP_FP32)


def build_task(k: dict, device):
    """Model, critic, objective, scheme, views and terms, on `device`."""
    configure(k["COMPILE"], k["COMPILE_MODE"], k["COMPILE_DYNAMIC"], k["COMPILE_TUNE"])
    model = build_autoencoder(k["ARCH"])

    # Seed weights from INIT_FROM unless a checkpoint of a resumed run supersedes them.
    resuming = k.get("RESUME") and latest(Path(k["RESUME"]) / "checkpoints")
    if k["INIT_FROM"] is not None and not resuming:
        report = load_partial(model, k["INIT_FROM"])
        refuse_frozen_missing(report, k["FREEZE"])

    scheme = build(k["GAN_SCHEME"], GAN_SCHEME)
    allow_donated_buffers(not scheme.retained_backward)
    critic = None
    if scheme.uses_critic:
        critic = build(k["CRITIC"], CRITIC)

    parts = AutoencoderParts(
        model=model,
        objective=_objective(k, model),
        scheme=scheme,
        critic=critic,
        views=[build(spec, LATENT_VIEW) for spec in k["LATENT_VIEWS"]],
        terms=_terms(k),
        teachers={
            name: build(s, TEACHER).freeze() for name, s in k["TEACHERS"].items()
        },
        aux_weights=k["AUX_WEIGHTS"],
        frozen=frozen_parts(model, k["FREEZE"]),
        lr_scales=k["LR_SCALES"],
        metrics_every=k["METRICS_EVERY"],
        usage_window=k["USAGE_WINDOW"],
    )
    task = TASK.get("autoencoder")(parts)

    # Memory/speed transforms, then move everything to the device.
    if k["GRAD_CKPT"]:
        apply_recompute(model)
    compile_parts((model.encoder, model.decoder, model.head))
    if k["COMPILE_LOSS"]:
        parts.objective.compile(**SETUP.kwargs())
        if critic is not None and not scheme.second_order:
            critic.compile(**SETUP.kwargs())
    layout = torch.channels_last if k["CHANNELS_LAST"] else torch.preserve_format
    task.memory_format = layout
    model.to(device, memory_format=layout)
    if critic is not None:
        critic.to(device, memory_format=layout)
    task.term_modules.to(device)
    _cast_to_policy(k, model, critic)

    # Bind kernels on the device, or on cpu when KERNELS is not auto.
    tuner.set_mode(k["KERNEL_TUNE"])
    bind_device = torch.device(device) if k["KERNELS"] == "auto" else "cpu"
    model.bind_kernels(bind_device)
    compute = build(k["PRECISION"], PRECISION).dtype
    task.bind_device(device, compute if k["LOSS_16BIT"] else None)
    return task


def build_bundles(task, k: dict, total_steps: int) -> dict[str, Bundle]:
    """One bundle per task group, each with its own scaler."""
    precision = build(k["PRECISION"], PRECISION)
    if precision.scaled and any(term.group != "model" for term, _ in task.terms):
        raise ValueError("latent terms need the model group under fp16")

    # 16-bit gradients still reduce in fp32 unless a wire dtype is asked for
    wire = WIRE_DTYPES[k["REDUCE_DTYPE"]] or (precision.weights and torch.float32)
    schedule_config = dict(k["SCHEDULER_CONFIG"])
    schedule_config["end"] = total_steps

    bundles = {}
    for name, groups in task.groups().items():
        # Per-group learning rates scale the base LR; the name labels the group's lr log
        params = [
            {
                "params": g["params"],
                "lr": k["LR"] * g["lr_scale"],
                "name": g.get("name", f"group{index}"),
            }
            for index, g in enumerate(groups)
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
            build(
                k["REDUCER"],
                REDUCER,
                params=[p for group in params for p in group["params"]],
                wire=wire,
            ),
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
