"""Knob validation: every area returns `list[str]`; nothing here needs a GPU.

Areas that depend on the model take it built on the meta device and ask its parts for the
capabilities they declare, never for registry names.
"""

from pathlib import Path

from latentdragon.gan import CRITIC, GAN_SCHEME
from latentdragon.latent import LATENT_TERM, LATENT_VIEW
from latentdragon.model.runtime.compile import SCOPES
from latentdragon.objectives import OBJECTIVE
from latentdragon.training.checkpoint import latest
from latentdragon.training.precision import PRECISION
from latentdragon_common.registry import Registry, build

SECOND_ORDER = ("r3gan", "stylegan2")
SCALED = ("fp16", "fp16-true")
TRUE_16 = ("bf16-true", "fp16-true")


def spec_name(spec) -> str | None:
    """The registry name inside a string or dict spec."""
    match spec:
        case str():
            return spec
        case {"name": str(name)}:
            return name
        case _:
            return None


def registry_problem(label: str, spec, registry: Registry) -> list[str]:
    """A named spec must resolve; a dotted path counts as resolved."""
    name = spec_name(spec)
    if name is None or name in registry or "." in name:
        return []
    return [f"{label}: unknown {name!r}; known: {', '.join(registry.names())}"]


def step_problems(k: dict) -> list[str]:
    """Batch/accumulation divisibility and the steps-or-epochs choice."""
    problems = []
    if k["GRAD_ACC"] < 1:
        problems.append("GRAD_ACC must be at least 1")
    elif k["BATCH_SIZE"] % k["GRAD_ACC"]:
        problems.append(
            f"BATCH_SIZE {k['BATCH_SIZE']} is not divisible by GRAD_ACC {k['GRAD_ACC']}"
        )
    if (k["MAX_STEPS"] <= 0) == (k["EPOCHS"] <= 0):
        problems.append("set exactly one of MAX_STEPS and EPOCHS")
    return problems


def precision_problems(k: dict) -> list[str]:
    """Precision settings that cannot work together.

    An fp16 wire is safe only with a loss scale: an overflow there is a non-finite gradient the
    scaler skips and backs off from. Without one, small gradients would flush to zero unseen.
    """
    problems = []
    if spec_name(k["REDUCER"]) == "none" and len(k["DEVICES"]) > 1:
        problems.append(
            "REDUCER none never averages; use flat or bucketed on several devices"
        )
    precision = spec_name(k["PRECISION"])
    if precision not in SCALED and k["REDUCE_DTYPE"] == "fp16":
        problems.append(
            f"PRECISION {precision} has no loss scale, so it cannot reduce on an fp16 wire"
        )
    if precision in TRUE_16 and spec_name(k["OPTIMIZER"]) != "adam16":
        problems.append(
            f"PRECISION {precision} holds 16-bit weights and needs OPTIMIZER adam16"
        )
    return problems


def component_problems(k: dict) -> list[str]:
    """Registry-named components that do not resolve."""
    problems = registry_problem("PRECISION", k["PRECISION"], PRECISION)
    problems += registry_problem("GAN_SCHEME", k["GAN_SCHEME"], GAN_SCHEME)
    problems += registry_problem("OBJECTIVE", k["OBJECTIVE"], OBJECTIVE)
    if k["CRITIC"] is not None:
        problems += registry_problem("CRITIC", k["CRITIC"], CRITIC)
    return problems


def gan_problems(k: dict, model) -> list[str]:
    """Scheme-specific problems and scheme/critic/freeze/head mismatches."""
    if spec_name(k["GAN_SCHEME"]) not in GAN_SCHEME:
        return []
    scheme = build(k["GAN_SCHEME"], GAN_SCHEME)
    problems = list(scheme.problems(k))
    if scheme.uses_critic and k["CRITIC"] is None:
        problems.append(f"GAN_SCHEME {scheme.name} needs CRITIC")
    if not scheme.uses_critic and k["CRITIC"] is not None:
        problems.append("CRITIC is set but GAN_SCHEME uses no critic")
    if scheme.uses_critic and _frozen(k, "head"):
        problems.append("a GAN scheme needs a trainable head; FREEZE holds it")
    if scheme.needs_anchor and not hasattr(model.head, "last_layer"):
        problems.append(
            f"GAN_SCHEME {scheme.name} weighs at the head's last_layer, which "
            f"{type(model.head).__name__} does not declare"
        )
    return problems


def _frozen(k: dict, part: str) -> bool:
    return any(p == part for p in k["FREEZE"])


def head_problems(k: dict, model) -> list[str]:
    """Head/objective pairing and pixel-input requirements."""
    head = type(model.head).__name__
    paired = getattr(model.head, "objective", None)
    objective = spec_name(k["OBJECTIVE"])
    problems = []
    if paired is None and objective is None:
        problems.append(f"{head} pairs with no objective; set OBJECTIVE")
    if paired is not None and objective is not None and objective != paired:
        problems.append(f"{head} pairs with OBJECTIVE {paired}, not {objective}")
    if getattr(model.head, "needs_pixels", False):
        problems.append(f"{head} needs pixels, and no pixel source is configured")
    return problems


def latent_problems(k: dict, model) -> list[str]:
    """Latent views and terms against domain, codebook, freeze and teacher knobs."""
    problems = []
    if k["PRECISION"] in SCALED:
        for spec in k["LATENT_TERMS"]:
            name = spec_name(spec["term"])
            if name in LATENT_TERM and LATENT_TERM.get(name).group != "model":
                problems.append(f"term {name} needs the model group under fp16")

    # Views: domain fit, codebook safety and freeze compatibility.
    quantized = getattr(model.latent, "codebook_size", 0) > 0
    domain = model.domain.name
    for spec in k["LATENT_VIEWS"]:
        issue = registry_problem("LATENT_VIEWS", spec, LATENT_VIEW)
        if issue:
            problems += issue
            continue
        view = LATENT_VIEW.get(spec_name(spec))
        name = spec_name(spec)
        if domain not in view.domains:
            problems.append(f"view {name} does not apply to {domain}")
        if quantized and not view.codebook_safe and not k["ALLOW_OFF_CODEBOOK"]:
            problems.append(
                f"view {name} leaves the codebook; set ALLOW_OFF_CODEBOOK to accept it"
            )
        if _frozen(k, "decoder") and _frozen(k, "head"):
            problems.append(f"view {name} cannot act with decoder and head frozen")

    # Terms: required parts must be trainable and required teachers present.
    for spec in k["LATENT_TERMS"]:
        issue = registry_problem("LATENT_TERMS", spec["term"], LATENT_TERM)
        if issue:
            problems += issue
            continue
        term = LATENT_TERM.get(spec_name(spec["term"]))
        kwargs = spec["term"] if isinstance(spec["term"], dict) else {}
        for part in term.parts_of(kwargs):
            if _frozen(k, part):
                problems.append(
                    f"term {spec_name(spec['term'])} needs {part}, which FREEZE holds"
                )
        if term.teacher is not None and term.teacher not in k["TEACHERS"]:
            problems.append(
                f"term {spec_name(spec['term'])} needs teacher {term.teacher}"
            )
    return problems


def stage_problems(k: dict) -> list[str]:
    """FREEZE and INIT_FROM consistency for staged training."""
    problems = []
    unfilled = k["INIT_FROM"] is None and (
        k["RESUME"] is None or latest(Path(k["RESUME"]) / "checkpoints") is None
    )
    if k["FREEZE"] and unfilled:
        problems.append("FREEZE without INIT_FROM trains on random frozen weights")
    if k["RESUME"] is None and k["INIT_FROM"] and not Path(k["INIT_FROM"]).exists():
        problems.append(f"INIT_FROM {k['INIT_FROM']} does not exist")
    return problems


def compile_problems(k: dict) -> list[str]:
    """Compile settings that cannot run."""
    problems = []
    if k["COMPILE"] not in SCOPES:
        problems.append(f"COMPILE {k['COMPILE']!r} is not one of {SCOPES}")
    if k["COMPILE_MODE"] == "reduce-overhead" and spec_name(k["REDUCER"]) == "bucketed":
        problems.append(
            "COMPILE_MODE reduce-overhead reuses memory that gradient views alias; "
            "set REDUCER flat"
        )
    return problems


def compile_notes(k: dict, model) -> list[str]:
    """Settings that run but cannot help, so a run starts knowing what it paid for."""
    notes = []
    if k["COMPILE"] == "function" and getattr(
        model.decoder, "function_scope_gaps", False
    ):
        notes.append(
            f"COMPILE function covers norms and SwiGLU only, which leaves "
            f"{type(model.decoder).__name__}'s convolutions eager; use module or model"
        )
    if k["COMPILE_LOSS"] and k["COMPILE"] == "off":
        notes.append(
            "COMPILE_LOSS with COMPILE off compiles the losses but not the model"
        )
    if k["COMPILE_LOSS"] and spec_name(k["GAN_SCHEME"]) in SECOND_ORDER:
        notes.append(
            f"GAN_SCHEME {spec_name(k['GAN_SCHEME'])} differentiates through a gradient, "
            "which a compiled graph cannot do, so its critic stays eager"
        )
    if k["COMPILE_LOSS"] and not k["LOSS_CONFIGS"]:
        notes.append("COMPILE_LOSS has no loss network to compile")
    if k["COMPILE"] in ("module", "model") and model.domain.name == "video":
        notes.append(
            "video chunks change shape; set COMPILE_DYNAMIC if clip length varies"
        )
    return notes


def data_problems(k: dict) -> list[str]:
    """A run needs at least one training source."""
    return [] if k["TRAIN_SOURCES"] else ["TRAIN_SOURCES is empty"]


def all_problems(k: dict, model) -> list[str]:
    """Every problem of every area; `model` is the arch built on the meta device."""
    return [
        *step_problems(k),
        *component_problems(k),
        *precision_problems(k),
        *gan_problems(k, model),
        *head_problems(k, model),
        *latent_problems(k, model),
        *stage_problems(k),
        *compile_problems(k),
        *data_problems(k),
    ]
