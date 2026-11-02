"""Autoencoder task: encode, views, decode, seeds, terms and the composed backward.

Every value is a slot dict; the task reads a part only through the contract in
`latentdragon.model.core.contracts` and the capabilities it names there.
"""

from dataclasses import dataclass, field

import torch
from sensecraft.metrics import psnr, ssim

from latentdragon.gan import GanCtx, GanMicro, GanScheme
from latentdragon.gan.scheme import detached, scaled
from latentdragon.latent import LatentCtx
from latentdragon.model.core import tree
from latentdragon.tasks.base import TASK, Task
from latentdragon.tasks.freeze import apply_freeze
from latentdragon.tasks.metrics import CodeUsage, NoUsage, ScalarEma
from latentdragon_common.calls import nothing

# Smoothed headline values under the names the research trainer logged
EMA_PAIRS = {
    "train/eloss": "train/nll",
    "train/psnr": "raw/psnr",
    "train/ssim": "raw/ssim",
}
# Sections whose scalars get no `<key>_ema` twin
UNTWINNED = (
    "perf/",
    "raw/",
    "val/",
    "latent_stat_",
    "optim/lr/",
    "optim/skipped/",
    "optim/loss_scale/",
)
FRAME_METRICS = {"psnr": psnr, "ssim": lambda a, b: ssim(a, b, 1.0)}


@dataclass
class AutoencoderParts:
    """Everything the task composes, resolved at build time."""

    model: torch.nn.Module
    objective: object
    scheme: GanScheme
    critic: torch.nn.Module | None = None
    views: list = field(default_factory=list)
    terms: list = field(default_factory=list)
    teachers: dict = field(default_factory=dict)
    aux_weights: dict = field(default_factory=dict)
    frozen: list = field(default_factory=list)
    lr_scales: dict = field(default_factory=dict)
    metrics_every: int = 1
    usage_window: float = 1000.0
    pixel_source: object = None


@TASK.register("autoencoder")
class AutoencoderTask(Task):
    """One selected model, objective, scheme, views and terms."""

    def __init__(self, parts: AutoencoderParts):
        # Parts and derived views of them
        self.parts = parts
        self.model = parts.model
        self.critic = parts.critic
        self.scheme = parts.scheme
        self.objective = parts.objective
        self.views = parts.views
        self.terms = parts.terms
        self.adv_views = [v for v in self.views if v.adversarial]
        self.plain_views = [v for v in self.views if not v.adversarial]
        self.term_modules = torch.nn.ModuleList([t for t, _ in self.terms])
        self._aux_weights = dict(parts.aux_weights)
        self.smooth = ScalarEma()
        self.usage = NoUsage()
        self._metric_every = max(parts.metrics_every, 1)

        # Freeze parts; skip autograd for frozen encoder, latent def and decoder
        apply_freeze(parts.frozen)
        frozen_ids = {id(p) for p in parts.frozen}
        encoder_held = id(self.model.encoder) in frozen_ids
        latent_held = encoder_held and id(self.model.latent) in frozen_ids
        self._encoder = no_grad_if(encoder_held, self.model.encoder)
        self._serialize = (
            self._held_serialize if latent_held else self.model.latent.serialize
        )
        self._aux_of = self._no_aux if latent_held else self._live_aux
        decoder_held = id(self.model.decoder) in frozen_ids
        self._decode = no_grad_if(decoder_held and latent_held, self.model.decode)

        # Head call, with the pixel source when the head wants one
        self._render = self.model.head
        if getattr(self.model.head, "needs_pixels", False):
            self._render = self._render_with_pixels

        if self.critic is not None:
            self.scheme = self.scheme.for_critic(self.critic)

    @torch.no_grad()
    def _held_serialize(self, slots, offset: int = 0):
        """Serialize with no gradient, for a frozen encoder and latent def."""
        encoded = self.model.latent.serialize(slots, offset)
        encoded.latent = detached(encoded.latent)
        return encoded

    def _encode(self, x: dict, offset: int):
        return self._serialize(self._encoder(x), offset)

    def _render_with_pixels(self, features):
        """Head call that also receives the source pixels."""
        return self.model.head(features, self.parts.pixel_source(features))

    def groups(self) -> dict[str, list[dict]]:
        """Trainable param groups: one per model part, the critic, then each term's own."""
        scales = self.parts.lr_scales

        def trainable(module):
            return [q for q in module.parameters() if q.requires_grad]

        # Model groups: encoder, latent def, decoder, head, each with its lr scale
        out = {"model": []}
        for name in ("encoder", "latent", "decoder", "head"):
            params = trainable(getattr(self.model, name))
            if params:
                group = {
                    "params": params,
                    "lr_scale": scales.get(name, 1.0),
                    "name": name,
                }
                out["model"].append(group)

        if self.critic is not None:
            out["critic"] = [
                {
                    "params": trainable(self.critic),
                    "lr_scale": scales.get("critic", 1.0),
                    "name": "critic",
                }
            ]

        for index, (term, _) in enumerate(self.terms):
            params = trainable(term)
            if params:
                group = {"params": params, "lr_scale": 1.0, "name": f"term{index}"}
                out.setdefault(term.group, []).append(group)
        return out

    def saved_modules(self) -> dict[str, torch.nn.Module]:
        """Modules written to checkpoints: model, plus critic and latent terms if any."""
        saved = {"model": self.model}
        if self.critic is not None:
            saved["critic"] = self.critic
        if self.terms:
            saved["latent_terms"] = self.term_modules
        return saved

    def summary_modules(self) -> dict[str, torch.nn.Module]:
        """The saved modules plus the frozen loss networks, which also take memory."""
        return {**self.saved_modules(), "loss": self.objective.loss}

    def ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def bind_device(self, device, loss_weights: torch.dtype | None = None) -> None:
        """Move the objective and teachers, open the code usage window."""
        self.objective.to(device, loss_weights)
        for teacher in self.parts.teachers.values():
            teacher.to(device)
        size = getattr(self.model.latent, "codebook_size", 0)
        if size:
            self.usage = CodeUsage(size, device, self.parts.usage_window)

    def train(self) -> None:
        """Train mode everywhere, then hold frozen parts back in eval."""
        self.model.train()
        if self.critic is not None:
            self.critic.train()
        self.term_modules.train()
        apply_freeze(self.parts.frozen)

    def eval(self) -> None:
        self.model.eval()
        self.model.latent.track_stats(True)

    def _live_aux(self, encoded) -> list:
        """Weighted `(weight, scalar)` latent losses."""
        weights = self._aux_weights
        return [(weights.get(k, 1.0), v) for k, v in encoded.losses.items()]

    @staticmethod
    def _no_aux(encoded) -> list:
        return []

    def _seeded_fake(self, x, latent, scale: float) -> tuple:
        """Decode `latent`; returns `(fake, d nll / d fake, nll, term logs)`."""
        fake = tree.map_tensors(lambda t: t.float(), self._render(self._decode(latent)))
        seed, nll, terms = self.objective.seeded(x, fake, scale)
        return fake, seed, nll, terms

    def _teacher_features(self, x) -> dict:
        """Features of `x` from every teacher, by name."""
        return {n: t.features(x) for n, t in self.parts.teachers.items()}

    def _term_aux(self, latent: LatentCtx) -> tuple[list, dict]:
        """Run the latent terms; returns their weighted losses and their logs."""
        last = getattr(self.model.encoder, "last_layer", None)
        anchor = last.weight if last is not None else None
        aux, logs = [], {}
        for term, weigh in self.terms:
            loss, term_logs = term(latent)
            aux.append((weigh(loss, latent, anchor), loss))
            logs.update(term_logs)
        return aux, logs

    def micro_step(self, ctx) -> dict:
        """Encode, decode, apply terms and views, run the GAN scheme."""
        x = ctx.batch
        self.model.latent.track_stats(ctx.step % self._metric_every == 0)

        # Encode and decode the main view
        with ctx.timer.phase("encode"):
            offset = ctx.step + ctx.micro_index * tree.batch_size(x)
            encoded = self._encode(x, offset)
        with ctx.timer.phase("decode"):
            fake, seed, nll, terms = self._seeded_fake(
                x, encoded.latent, ctx.seed_scale
            )

        # Latent terms and the latent losses they join
        latent = LatentCtx(x, encoded, ctx.step, fake=fake, seed=seed)
        if self.terms:
            latent.teacher = self._teacher_features(x)
        term_aux, term_logs = self._term_aux(latent) if self.terms else ([], {})
        aux = self._aux_of(encoded) + term_aux
        micros = [GanMicro(x, fake, seed, aux)]

        # Extra views, then the composed backward through the scheme
        with ctx.timer.phase("views"):
            self._run_views(x, encoded.latent, ctx, micros)
        with ctx.timer.phase("scheme"):
            logs = self._run_scheme(ctx, micros)

        # Collect metrics
        # Sections: train/ as the research trainer named them, gan/ for the scheme, the
        # latent def's own sections for its logs, raw/ for values smoothed elsewhere
        metrics = {"train/nll": nll.detach()}
        metrics.update({f"train/{k}": v for k, v in terms.items()})
        metrics.update({f"gan/{k}": v for k, v in logs.items()})
        metrics.update({f"train/{k}": v.detach() for k, v in encoded.losses.items()})
        metrics.update(encoded.logs)
        metrics.update(term_logs)
        metrics.update(self._usage(encoded, ctx))
        metrics.update(self._quality(x, fake, encoded, ctx.step, ctx.micro_index))
        return metrics

    def _run_views(self, x, latent, ctx, micros: list) -> None:
        """Decode every view; critic-visible ones join `micros`, others join aux."""
        for view in self.plain_views:
            x_view, latent_view = view(x, latent, ctx.step)
            fake, seed, _, _ = self._seeded_fake(x_view, latent_view, ctx.seed_scale)
            micros[0].aux.append((view.weight, tree.dot(fake, detached(seed))))
        for view in self.adv_views:
            x_view, latent_view = view(x, latent, ctx.step)
            fake, seed, _, _ = self._seeded_fake(x_view, latent_view, ctx.seed_scale)
            micros.append(GanMicro(x_view, fake, scaled(seed, view.weight)))

    def _run_scheme(self, ctx, micros: list) -> dict:
        """Call the active or idle scheme policy once per micro (view)."""
        count = len(micros)
        policy = (
            self.scheme.active if ctx.step >= self.scheme.start else self.scheme.idle
        )
        backward_d = ctx.backward.get("critic", nothing)
        backward_g = ctx.backward["model"]
        pending: list = []
        if count > 1:
            self._share_across_views(micros, count)
            backward_g = _Deferred(pending)
        last = getattr(self.model.head, "last_layer", None)
        anchor = last.weight if last is not None else None
        logs: dict = {}
        for index, micro in enumerate(micros):
            logs.update(
                policy(
                    GanCtx(
                        m=micro,
                        micro_index=ctx.micro_index * count + index,
                        micro_count=ctx.micro_count * count,
                        critic=self.critic,
                        backward_g=backward_g,
                        backward_d=backward_d,
                        anchor=anchor,
                        sync=ctx.sync,
                        seed_scale=ctx.seed_scale,
                    )
                )
            )

        # Views share the encoder's graph, so they take one backward together
        if pending:
            final = ctx.micro_index == ctx.micro_count - 1
            ctx.backward["model"](sum(pending), final=final)
        return logs

    @staticmethod
    def _share_across_views(micros: list, count: int) -> None:
        """Scale seeds and aux weights by `count` so views jointly average."""
        for micro in micros:
            micro.recon_seed = scaled(micro.recon_seed, count)
            micro.aux = [(weight * count, value) for weight, value in micro.aux]

    def _usage(self, encoded, ctx) -> dict:
        """The code-usage window, closed on the last micro of a metrics step."""
        codes = getattr(self.model.latent, "codes", None)
        if codes is None or ctx.step % self._metric_every != 0:
            return {}
        self.usage.mark(codes(encoded))
        if ctx.micro_index != ctx.micro_count - 1:
            return {}
        return self.usage.advance(self._metric_every, ctx.sync)

    def _quality(self, x, fake, encoded, step: int, micro_index: int) -> dict:
        """Frame metrics per data slot and latent stats; first micro of a metrics step."""
        if micro_index != 0 or step % self._metric_every != 0:
            return {}
        domain = self.model.domain
        single = len(domain.metrics) == 1
        out = {}
        with torch.autocast(tree.leaves(x)[0].device.type, enabled=False):
            # Frames mapped from [-1, 1] to [0, 1], in fp32
            for slot, names in domain.metrics.items():
                frames = domain.to_frames[slot]
                flat_fake = frames(fake[slot].detach().float()) * 0.5 + 0.5
                flat_real = frames(x[slot].float()) * 0.5 + 0.5
                for name in names:
                    if name in FRAME_METRICS:
                        key = f"raw/{name}" if single else f"raw/{slot}/{name}"
                        out[key] = FRAME_METRICS[name](flat_fake, flat_real).mean()
            out.update(latent_stats(self.model.latent.raw(encoded)))
        return out

    def smooth_metrics(self, step: int, metrics: dict) -> dict:
        """Add the smoothed headline scalars and an `_ema` twin of every other scalar."""
        out = dict(metrics)
        for name, source in EMA_PAIRS.items():
            if source in metrics:
                out[name] = self.smooth.update(step, name, metrics[source])

        # Every other scalar keeps its raw value and gains an EMA twin
        smoothed = {*EMA_PAIRS, *EMA_PAIRS.values()}
        raw = {
            f"{key}_ema": value
            for key, value in metrics.items()
            if not key.startswith(UNTWINNED)
            and key not in smoothed
            and not key.endswith("_ema")
            and isinstance(value, torch.Tensor)
        }
        out.update(self.smooth.update_all(step, raw))
        return out

    def post_step(self, step: int, sync) -> None:
        """Share a stat-tracking critic's buffers from rank 0."""
        if self.critic is not None and self.critic.tracks_stats:
            for buffer in self.critic.buffers():
                sync.broadcast_(buffer)

    @torch.no_grad()
    def reconstruct(self, x: dict, variant=None) -> tuple:
        """Deterministic reconstruction; `variant` is one of the latent def's eval variants."""
        encoded = self.model.encode(x)
        latent = encoded.latent
        if variant is not None:
            latent = self.model.latent.variant(latent, variant)
        return self.model.head(self.model.decode(latent)), encoded


def latent_stats(raw: dict) -> dict:
    """Mean and std of every float latent tensor, and per channel along dim 1.

    One tensor keeps the research trainer's names (`train/z_mean`, `latent_stat_mean/ch00`);
    several get their slot path in the per-channel key. Per-channel values stay one vector per
    stat, which the metric log writes as `.../ch00`, `.../ch01`, ...
    """
    found = {
        path: value.detach().float()
        for path, value in tree.named_tensors(raw).items()
        if value.is_floating_point()
    }
    out = {}
    for path, value in found.items():
        out[f"train/{path}_mean"] = value.mean()
        out[f"train/{path}_std"] = value.std()
        if value.ndim < 2:
            continue
        section = "" if len(found) == 1 else f"{path}/"
        channels = value.transpose(0, 1).flatten(1)
        out[f"latent_stat_mean/{section}ch"] = channels.mean(1)
        out[f"latent_stat_std/{section}ch"] = channels.std(1)
    return out


class _Deferred:
    """Stands in for a generator backward; the task sums what it collects."""

    def __init__(self, pending: list):
        self.pending = pending

    def __call__(self, scalar: torch.Tensor, final: bool = False) -> None:
        self.pending.append(scalar)


def no_grad_if(held: bool, call):
    """`call` under no_grad when it belongs to a held part."""
    return torch.no_grad()(call) if held else call
