"""Autoencoder task: encode, bottleneck, views, decode, seeds, terms, composed backward."""

import functools
from dataclasses import dataclass, field

import torch
from sensecraft.metrics import psnr, ssim

from latentdragon.gan import GanCtx, GanMicro, GanScheme, seed_of
from latentdragon.latent import LatentCtx
from latentdragon.tasks.base import TASK, Task
from latentdragon.tasks.freeze import apply_freeze
from latentdragon.tasks.metrics import CodeUsage, NoUsage, ScalarEma
from latentdragon_common.calls import nothing

EMA_KEYS = {
    "vq/loss": "aux/vq",
    "vq/usage": "raw/usage",
    "vq/perplexity": "raw/perplexity",
    "vq/perplexity_rate": "raw/perplexity_rate",
}
DEFAULT_AUX_WEIGHTS = {"vq": 1.0, "kl": 0.0}
DIRECT_KEYS = ("commitment_loss", "codebook_loss", "ivq_loss")


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
    enc_lr_scale: float = 1.0
    vq_lr_scale: float = 1.0
    disc_lr_scale: float = 1.0
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
        self._aux_weights = {**DEFAULT_AUX_WEIGHTS, **parts.aux_weights}
        self.smooth = ScalarEma()
        self.usage = NoUsage()
        self._metric_every = max(parts.metrics_every, 1)

        # Freeze parts; skip autograd for frozen encoder / bottleneck / trunk
        apply_freeze(parts.frozen)
        frozen_ids = {id(p) for p in parts.frozen}
        encoder_held = id(self.model.encoder) in frozen_ids
        latent_held = encoder_held and id(self.model.bottleneck) in frozen_ids
        self._encode = no_grad_if(encoder_held, self.model.encode)
        self._bottleneck = self.model.bottleneck
        if latent_held:
            self._bottleneck = self._held_bottleneck
        self._aux_of = self._no_aux if latent_held else self._live_aux
        trunk_held = id(self.model.trunk) in frozen_ids
        self._decode = no_grad_if(trunk_held and latent_held, self.model.decode)

        # Head call, with the pixel source when the head wants one
        self._render = self.model.head
        if self.model.head.needs_pixels:
            self._render = self._render_with_pixels

        if self.critic is not None:
            self.scheme = self.scheme.for_critic(self.critic)

    @torch.no_grad()
    def _held_bottleneck(self, raw, offset: int = 0):
        """Bottleneck pass with no gradient, for a frozen encoder and bottleneck."""
        out = self.model.bottleneck(raw, offset)
        out.z = out.z.detach()
        return out

    def _render_with_pixels(self, features):
        """Head call that also receives the source pixels."""
        return self.model.head(features, self.parts.pixel_source(features))

    def groups(self) -> dict[str, list[dict]]:
        """Trainable param groups: model parts, critic, then each term's own group."""
        model = self.model
        p = self.parts

        def trainable(module):
            return [q for q in module.parameters() if q.requires_grad]

        # Model groups: encoder, trunk + head, bottleneck, each with its lr scale
        spec = [
            (trainable(model.encoder), p.enc_lr_scale),
            (trainable(model.trunk) + trainable(model.head), 1.0),
            (trainable(model.bottleneck), p.vq_lr_scale),
        ]
        out = {
            "model": [
                {"params": params, "lr_scale": scale}
                for params, scale in spec
                if params
            ]
        }

        if self.critic is not None:
            out["critic"] = [
                {"params": trainable(self.critic), "lr_scale": p.disc_lr_scale}
            ]

        for term, _ in self.terms:
            params = trainable(term)
            if params:
                group = {"params": params, "lr_scale": 1.0}
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

    def ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def bind_device(self, device) -> None:
        """Bind kernels, move the objective and teachers, open the usage window."""
        self.objective.to(device)
        for teacher in self.parts.teachers.values():
            teacher.to(device)
        size = getattr(self.model.bottleneck, "codebook_size", 0)
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

    def _live_aux(self, out) -> list:
        """Weighted `(weight, scalar)` aux losses of a bottleneck output."""
        weights = self._aux_weights
        return [(weights.get(k, 1.0), v) for k, v in out.aux.items()]

    @staticmethod
    def _no_aux(out) -> list:
        return []

    def _seeded_fake(self, x, z, scale: float) -> tuple:
        """Decode `z`; returns `(fake, d nll / d fake, nll, term logs)`."""
        fake = self._render(self._decode(z)).float()
        nll, terms = self.objective(x, fake)
        seed = seed_of(nll, fake, scale, retain_graph=True)
        return fake, seed, nll, terms

    def _teacher_features(self, x) -> dict:
        """Features of `x` from every teacher, by name."""
        return {n: t.features(x) for n, t in self.parts.teachers.items()}

    def _term_aux(self, latent: LatentCtx) -> tuple[list, dict]:
        """Run the latent terms; returns their weighted losses and their logs."""
        anchor = self.model.encoder.out_layer.weight
        aux, logs = [], {}
        for term, weigh in self.terms:
            loss, term_logs = term(latent)
            aux.append((float(weigh(loss, latent, anchor)), loss))
            logs.update(term_logs)
        return aux, logs

    def micro_step(self, ctx) -> dict:
        """Encode, quantize, decode, apply terms and views, run the GAN scheme."""
        x = ctx.batch

        # Encode and decode the main view
        with ctx.timer.phase("encode"):
            raw = self._encode(x)
            out = self._bottleneck(raw, ctx.step + ctx.micro_index * x.shape[0])
        with ctx.timer.phase("decode"):
            fake, seed, nll, terms = self._seeded_fake(x, out.z, ctx.seed_scale)

        # Latent terms and the aux losses they join
        latent = LatentCtx(x, out.raw, out.z, ctx.step, fake=fake, seed=seed)
        if self.terms:
            latent.teacher = self._teacher_features(x)
        term_aux, term_logs = self._term_aux(latent) if self.terms else ([], {})
        aux = self._aux_of(out) + term_aux
        micros = [GanMicro(x, fake, seed, aux)]

        # Extra views, then the composed backward through the scheme
        with ctx.timer.phase("views"):
            self._run_views(x, out.z, ctx, micros)
        with ctx.timer.phase("scheme"):
            logs = self._run_scheme(ctx, micros)

        # Collect metrics
        metrics = {"train/nll": nll.detach()}
        metrics.update({f"train/{k}": v for k, v in terms.items()})
        metrics.update({f"train/{k}": v for k, v in logs.items()})
        metrics.update({f"aux/{k}": v.detach() for k, v in out.aux.items()})
        if "kl" in out.aux:
            metrics["train/kl"] = out.aux["kl"].detach()
        metrics.update(term_logs)
        metrics.update(self._bottleneck_metrics(out, ctx))
        metrics.update(self._quality(x, fake, out.raw, ctx.step, ctx.micro_index))
        return metrics

    def _run_views(self, x, z, ctx, micros: list) -> None:
        """Decode every view; critic-visible ones join `micros`, others join aux."""
        for view in self.plain_views:
            x_view, z_view = view(x, z, ctx.step)
            fake, seed, _, _ = self._seeded_fake(x_view, z_view, ctx.seed_scale)
            micros[0].aux.append((view.weight, (fake * seed.detach()).sum()))
        for view in self.adv_views:
            x_view, z_view = view(x, z, ctx.step)
            fake, seed, _, _ = self._seeded_fake(x_view, z_view, ctx.seed_scale)
            micros.append(GanMicro(x_view, fake, seed * view.weight))

    def _run_scheme(self, ctx, micros: list) -> dict:
        """Call the active or idle scheme policy once per micro (view)."""
        count = len(micros)
        policy = (
            self.scheme.active if ctx.step >= self.scheme.start else self.scheme.idle
        )
        backward_d = ctx.backward.get("critic", nothing)
        backward_g = ctx.backward["model"]
        if count > 1:
            self._share_across_views(micros, count)
            backward_g = functools.partial(backward_g, retain_graph=True)
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
                        anchor=self.model.head.anchor,
                        sync=ctx.sync,
                        seed_scale=ctx.seed_scale,
                    )
                )
            )
        return logs

    @staticmethod
    def _share_across_views(micros: list, count: int) -> None:
        """Scale seeds and aux weights by `count` so views jointly average."""
        for micro in micros:
            micro.recon_seed = micro.recon_seed * count
            micro.aux = [(weight * count, value) for weight, value in micro.aux]

    def _bottleneck_metrics(self, out, ctx) -> dict:
        """Bottleneck logs, plus the code-usage window on the last micro of a step."""
        logs = {
            (f"vq/{k}" if k in DIRECT_KEYS else f"raw/{k}"): v
            for k, v in out.metrics.items()
        }
        if out.indices is not None and ctx.step % self._metric_every == 0:
            self.usage.mark(out.indices)
            if ctx.micro_index == ctx.micro_count - 1:
                logs.update(self.usage.advance(self._metric_every, ctx.sync))
        return logs

    def _quality(self, x, fake, raw, step: int, micro_index: int) -> dict:
        """PSNR, SSIM and latent stats; first micro of every metrics step only."""
        if micro_index != 0 or step % self._metric_every != 0:
            return {}
        # Frames mapped from [-1, 1] to [0, 1], in fp32
        with torch.autocast(x.device.type, enabled=False):
            flat_fake = self._frames(fake.detach().float()) * 0.5 + 0.5
            flat_real = self._frames(x.float()) * 0.5 + 0.5
            return {
                "quality/psnr": psnr(flat_fake, flat_real).mean(),
                "quality/ssim": ssim(flat_fake, flat_real, 1.0).mean(),
                "train/z_mean": raw.detach().float().mean(),
                "train/z_std": raw.detach().float().std(),
            }

    def _frames(self, signal: torch.Tensor) -> torch.Tensor:
        """A batch of 2D frames for image metrics."""
        return self.model.domain.to_frames(signal)

    def smooth_metrics(self, step: int, metrics: dict) -> dict:
        """Add the EMA-smoothed headline scalars under their stable names."""
        out = dict(metrics)
        pairs = {
            "train/eloss": "train/nll",
            "train/psnr": "quality/psnr",
            "train/ssim": "quality/ssim",
            **EMA_KEYS,
        }
        for name, source in pairs.items():
            if source in metrics:
                out[name] = self.smooth.update(step, name, metrics[source])
        return out

    def post_step(self, step: int, sync) -> None:
        """Share a stat-tracking critic's buffers from rank 0."""
        if self.critic is not None and self.critic.tracks_stats:
            for buffer in self.critic.buffers():
                sync.broadcast_(buffer)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, keep_channels: int | None = None) -> tuple:
        """Deterministic recon; `keep_channels` zeroes latent channels past that count."""
        out = self.model.bottleneck(self.model.encode(x))
        z = out.z if keep_channels is None else zero_tail(out.z, keep_channels)
        return self.model.head(self.model.decode(z)), out


def no_grad_if(held: bool, call):
    """`call` under no_grad when it belongs to a held part."""
    return torch.no_grad()(call) if held else call


def zero_tail(z: torch.Tensor, keep: int) -> torch.Tensor:
    """`z` with channels from index `keep` on zeroed."""
    kept = z.clone()
    kept[:, keep:] = 0
    return kept
