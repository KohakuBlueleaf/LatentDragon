"""LatentDragon autoencoder for `transformers` and `diffusers` remote loading.

Shipped next to the weights. It needs only `torch` and `transformers`:

    from transformers import AutoModel            # or: from diffusers import AutoModel
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

It reimplements, for inference, the part of LatentDragon these exports use: LatentMaid image
encoder and decoder (`mlp` blocks, no attention), a `vq` latent on a projected ("kohaku")
codebook, optionally `nested` by partition, and a `conv` head. Module names match
LatentDragon's, so the same `model.safetensors` loads into both. An arch outside that set is
refused at build; LatentDragon itself still loads it (`latentdragon.io.hf.from_pretrained`).

Pixels are RGB in [-1, 1], `(B, 3, H, W)`, with H and W multiples of `config.spatial_scale`.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_latentdragon import LatentDragonConfig

ACTS = {"silu": F.silu, "gelu": F.gelu, "mish": F.mish}
SEARCH_ROWS = 4096  # latent positions per nearest-code search chunk


def _spec(value) -> tuple[str, dict]:
    """A registry choice (`"name"` or `{"name": ..., **kwargs}`) as name and kwargs."""
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict):
        kwargs = dict(value)
        return kwargs.pop("name"), kwargs
    raise ValueError(f"unsupported spec {value!r}")


def _wide(dtype: torch.dtype) -> torch.dtype:
    """`dtype`, at least fp32."""
    return torch.promote_types(dtype, torch.float32)


# ---------------------------------------------------------------- layers


class ChannelNorm(nn.Module):
    """RMS or layer norm over the channel axis at dim 1, eps 1e-6."""

    def __init__(self, dim: int, kind: str):
        super().__init__()
        if kind not in ("rms", "layer"):
            raise ValueError(f"norm {kind!r} is not supported here")
        self.kind = kind
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if kind == "layer" else None

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, -1)
        shape = (x.shape[-1],)
        if self.kind == "rms":
            x = F.rms_norm(x, shape, self.weight, 1e-6)
        else:
            x = F.layer_norm(x, shape, self.weight, self.bias, 1e-6)
        return x.transpose(1, -1)


def make_norm(kind: str, dim: int) -> nn.Module:
    """A head or block norm by its LatentDragon name."""
    if kind == "group32":
        return nn.GroupNorm(32, dim, eps=1e-6)
    return ChannelNorm(dim, kind)


class SwiGLU(nn.Module):
    """`out(silu(a) * gate)` with `a, gate` the halves of `inp(x)`, on channels-last input."""

    def __init__(self, in_dim: int, mlp_dim: int, out_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, mlp_dim * 2)
        self.out = nn.Linear(mlp_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.inp(x).chunk(2, dim=-1)
        return self.out(F.silu(x) * gate)


class MLPBlock(nn.Module):
    """LatentDragon's `mlp` block: norm, depthwise 3x3, channel MLP, residual."""

    def __init__(self, dim: int, mlp_dim: int, norm: str):
        super().__init__()
        self.norm = make_norm(norm, dim)
        self.mixer = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.swiglu = SwiGLU(dim, mlp_dim, dim) if mlp_dim > 0 else nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.mixer(self.norm(x)).transpose(1, -1)
        return x + self.swiglu(mixed).transpose(1, -1)


def _check_block(block, attn) -> None:
    """Only plain `mlp` blocks and no attention slots are reimplemented here."""
    name, kwargs = _spec(block)
    if name != "mlp" or kwargs.get("unit_rows"):
        raise ValueError(f"block {block!r} is not supported here; use LatentDragon")
    if attn and any(attn):
        raise ValueError("attention slots are not supported here; use LatentDragon")


def _blocks(count: int, dim: int, mlp: int, norm: str) -> nn.ModuleList:
    return nn.ModuleList(MLPBlock(dim, mlp, norm) for _ in range(count))


# ---------------------------------------------------------------- encoder, decoder, head


class Encoder(nn.Module):
    """LatentMaid image encoder: pixel-unshuffle stages of `mlp` blocks."""

    def __init__(
        self,
        out_channels: int,
        stages: list,
        in_channels: int = 3,
        attn=None,
        norm: str = "rms",
        block="mlp",
        **_,
    ):
        super().__init__()
        _check_block(block, attn)
        self.scales = [scale for scale, *_ in stages]
        projs, blocks = [], []
        dim = in_channels
        for scale, hidden, mlp, count in stages:
            dim = dim * scale**2
            hidden = hidden or dim
            projs.append(nn.Conv2d(dim, hidden, 1) if hidden != dim else nn.Identity())
            blocks.append(_blocks(count, hidden, mlp, norm))
            dim = hidden
        self.stage_projs = nn.ModuleList(projs)
        self.stages = nn.ModuleList(blocks)
        self.out_proj = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        for scale, proj, stage in zip(self.scales, self.stage_projs, self.stages):
            x = proj(F.pixel_unshuffle(x, scale))
            for block in stage:
                x = block(x)
        return self.out_proj(x)


class Decoder(nn.Module):
    """LatentMaid image decoder: pixel-shuffle stages, the last stage spec runs first."""

    def __init__(
        self,
        in_channels: int,
        stages: list,
        attn=None,
        norm: str = "rms",
        block="mlp",
        **_,
    ):
        super().__init__()
        _check_block(block, attn)
        walk = list(reversed(stages))
        self.scales = [scale for scale, *_ in walk]
        projs, blocks = [], []
        dim = in_channels
        for scale, hidden, mlp, count in walk:
            projs.append(nn.Conv2d(dim, hidden, 1) if dim != hidden else nn.Identity())
            blocks.append(_blocks(count, hidden, mlp, norm))
            dim = hidden // scale**2
        self.stage_projs = nn.ModuleList(projs)
        self.stages = nn.ModuleList(blocks)
        self.out_channels = dim

    def forward(self, z: Tensor) -> Tensor:
        x = z
        for scale, proj, stage in zip(self.scales, self.stage_projs, self.stages):
            x = proj(x)
            for block in stage:
                x = block(x)
            x = F.pixel_shuffle(x, scale)
        return x


class StandardizedConv2d(nn.Conv2d):
    """A conv whose weight rows are zero-mean and unit-norm at every call."""

    def _conv_forward(self, x, weight, bias):
        rows = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
        norm = rows.flatten(1).norm(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return super()._conv_forward(x, rows / norm, bias)


class Head(nn.Module):
    """LatentDragon's `conv` head: optional norm and activation, then a projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        kernel_size: int = 1,
        norm: str | None = None,
        act: str | None = None,
        video: bool = False,
        standardize: bool = False,
        fp32: bool = False,
        **_,
    ):
        super().__init__()
        if video:
            raise ValueError("video heads are not supported here; use LatentDragon")
        if act not in (None, "silu", "mish"):
            raise ValueError(f"head act {act!r} is not supported here")
        self.fp32 = fp32
        self.act = act
        self.norm = make_norm(norm, in_channels) if norm else nn.Identity()
        conv = StandardizedConv2d if standardize else nn.Conv2d
        self.proj = conv(in_channels, out_channels, kernel_size, 1, kernel_size // 2)

    def _project(self, features: Tensor) -> Tensor:
        features = self.norm(features)
        if self.act:
            features = ACTS[self.act](features)
        return self.proj(features)

    def forward(self, features: Tensor) -> Tensor:
        if not self.fp32:
            return self._project(features)
        with torch.autocast(features.device.type, enabled=False):
            return self._project(features.to(self.proj.weight.dtype))


# ---------------------------------------------------------------- codebook and quantizer


def _sharing(mode: str, separated_proj: bool, partitions: int, levels: int, dim: int):
    """`(tables, groups, out_dim, layout)` of a projected codebook's partition sharing."""
    match mode, separated_proj:
        case "shared", _:
            return 1, levels, dim, "shared"
        case "indep", _:
            return partitions, levels * partitions, dim, "separated"
        case "hyper", True:
            return 1, levels * partitions, dim, "separated"
        case "hyper", False:
            return 1, levels, partitions * dim, "joint"
    raise ValueError(f"partition_mode {mode!r} is not supported")


class ProjectedCodebook(nn.Module):
    """`scale * proj(base)` as an `(R, P, N, D)` grid; KohakuVQ's projected codebook."""

    def __init__(
        self,
        n_embed: int,
        e_dim: int,
        partitions: int = 1,
        rvq_levels: int = 1,
        partition_mode: str = "shared",
        separated_proj: bool = True,
        mlp_projector: bool = False,
        mlp_hidden_dim: int = 1024,
        mlp_act: str = "silu",
        base_dim: int | None = None,
        trainable_base: bool = False,
        dim_scale: bool = False,
    ):
        super().__init__()
        self.n_embed, self.e_dim = n_embed, e_dim
        self.partitions, self.rvq_levels = partitions, rvq_levels
        self.mode = partition_mode
        self.tables, self.groups, out_dim, self.layout = _sharing(
            partition_mode, separated_proj, partitions, rvq_levels, e_dim
        )
        self.mlp_act = mlp_act
        self.mlp_projector = mlp_projector
        base_dim = e_dim if base_dim is None else base_dim

        head = (self.tables,) if partition_mode == "indep" else ()
        base = torch.zeros(*head, n_embed, base_dim)
        if trainable_base:
            self.codebook_base = nn.Parameter(base)
        else:
            self.register_buffer("codebook_base", base, persistent=True)

        groups = self.groups
        if mlp_projector:
            hidden = mlp_hidden_dim
            self.codebook_proj_w1 = nn.Parameter(torch.zeros(groups, base_dim, hidden))
            self.codebook_proj_b1 = nn.Parameter(torch.zeros(groups, 1, hidden))
            self.codebook_proj_w2 = nn.Parameter(torch.zeros(groups, hidden, out_dim))
            self.codebook_proj_b2 = nn.Parameter(torch.zeros(groups, 1, out_dim))
        else:
            self.codebook_proj_w = nn.Parameter(torch.zeros(groups, base_dim, out_dim))
            self.codebook_proj_b = nn.Parameter(torch.zeros(groups, 1, out_dim))
        self.codebook_scale = nn.Parameter(torch.ones(rvq_levels))
        if dim_scale:
            self.codebook_dim_scale = nn.Parameter(torch.ones(rvq_levels * e_dim))
        else:
            self.codebook_dim_scale = None
        self._cache = None

    def _load_from_state_dict(self, *args, **kwargs):
        self._cache = None
        super()._load_from_state_dict(*args, **kwargs)

    def _build(self) -> Tensor:
        """The full grid, computed in at least fp32."""
        dtype = _wide(self.codebook_base.dtype)
        base = self.codebook_base.to(dtype)
        n_embed, base_dim = base.shape[-2:]
        tabled = base.reshape(self.tables, n_embed, base_dim).unsqueeze(0)
        repeats = self.groups // self.tables
        base = tabled.expand(repeats, -1, -1, -1).reshape(self.groups, n_embed, -1)

        with torch.autocast(base.device.type, enabled=False):
            if self.mlp_projector:
                hidden = torch.bmm(base, self.codebook_proj_w1.to(dtype))
                hidden = ACTS[self.mlp_act](hidden + self.codebook_proj_b1.to(dtype))
                codes = torch.bmm(hidden, self.codebook_proj_w2.to(dtype))
                codes = codes + self.codebook_proj_b2.to(dtype)
            else:
                codes = torch.bmm(base, self.codebook_proj_w.to(dtype))
                codes = codes + self.codebook_proj_b.to(dtype)

        levels, parts, dim = self.rvq_levels, self.partitions, self.e_dim
        match self.layout:
            case "shared":
                grid = codes.unsqueeze(1).expand(-1, parts, -1, -1)
            case "separated":
                grid = codes.reshape(levels, parts, n_embed, dim)
            case "joint":
                grid = codes.reshape(levels, n_embed, parts, dim).permute(0, 2, 1, 3)
        grid = grid * self.codebook_scale.to(dtype).reshape(-1, 1, 1, 1)
        if self.codebook_dim_scale is not None:
            dims = self.codebook_dim_scale.to(dtype).reshape(levels, 1, 1, dim)
            grid = grid * dims
        return grid

    def forward(self) -> Tensor:
        """The `(R, P, N, D)` grid; cached while neither training nor recording grads."""
        if self.training or torch.is_grad_enabled():
            return self._build()
        device = self.codebook_base.device
        if self._cache is None or self._cache.device != device:
            self._cache = self._build()
        return self._cache


class Quantizer(nn.Module):
    """Nearest-code search over P partitions and R residual levels."""

    def __init__(self, emb_dim: int, n_embed: int, partitions: int = 1, **config):
        super().__init__()
        if emb_dim % partitions:
            raise ValueError(f"dim {emb_dim} not divisible by partitions {partitions}")
        for unused in ("beta", "ivq_loss_weight"):
            config.pop(unused, None)
        rvq_levels = config.pop("rvq_levels", 1)
        self.partitions, self.rvq_levels = partitions, rvq_levels
        self.n_embed, self.e_dim = n_embed, emb_dim // partitions
        self.codebook = ProjectedCodebook(
            n_embed, self.e_dim, partitions, rvq_levels, **config
        )

    def _nearest(self, z: Tensor, level: Tensor) -> tuple[Tensor, Tensor]:
        """`z (M, P, D)` against `level (P, N, D)`: codes `(M, P, D)` and indices `(M, P)`."""
        dtype = _wide(z.dtype)
        codes = torch.empty_like(z)
        indices = torch.empty(z.shape[:2], dtype=torch.long, device=z.device)
        for part in range(self.partitions):
            table = level[part].to(dtype)
            table_sq = table.pow(2).sum(dim=1)
            for start in range(0, z.shape[0], SEARCH_ROWS):
                rows = z[start : start + SEARCH_ROWS, part].to(dtype)
                dists = (
                    rows.pow(2).sum(dim=1, keepdim=True)
                    + table_sq[None]
                    - 2 * rows @ table.T
                )
                best = dists.argmin(dim=1)
                indices[start : start + SEARCH_ROWS, part] = best
                codes[start : start + SEARCH_ROWS, part] = table[best].to(z.dtype)
        return codes, indices

    @torch.no_grad()
    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Channels-last `(..., C)` -> codes `(..., C)` and local indices `(..., P, R)`."""
        head = z.shape[:-1]
        residual = z.reshape(-1, self.partitions, self.e_dim)
        grid = self.codebook()
        total = torch.zeros_like(residual)
        found = []
        with torch.autocast(z.device.type, enabled=False):
            for level in grid.unbind(0):
                codes, indices = self._nearest(residual, level)
                total = total + codes
                residual = residual - codes
                found.append(indices)
        indices = torch.stack(found, dim=-1)
        codes = total.reshape(*head, self.partitions * self.e_dim)
        return codes, indices.reshape(*head, self.partitions, self.rvq_levels)

    def lookup(self, indices: Tensor) -> Tensor:
        """Local indices `(..., P, R)` -> summed codes `(..., C)`."""
        grid = self.codebook()
        levels = torch.arange(self.rvq_levels, device=indices.device)
        parts = torch.arange(self.partitions, device=indices.device)[:, None]
        codes = grid[levels, parts, indices].sum(dim=-2)
        return codes.reshape(*indices.shape[:-2], self.partitions * self.e_dim)


class VQLatent(nn.Module):
    """LatentDragon's `vq` latent def with a `kohaku` quantizer, for inference."""

    def __init__(self, dim: int, quantizer, normalize: str | None = None, **_):
        super().__init__()
        name, config = _spec(quantizer)
        if name != "kohaku":
            raise ValueError(
                f"quantizer {name!r} is not supported here; use LatentDragon"
            )
        if normalize not in (None, "rms"):
            raise ValueError(f"normalize {normalize!r} is not supported")
        self.normalize = normalize
        self.vq = Quantizer(dim, **config)

    def channels_last(self, latents: Tensor) -> Tensor:
        """Channels-last encoder output, each partition on unit RMS when `normalize` is set."""
        last = latents.movedim(1, -1)
        if self.normalize is None:
            return last
        vq = self.vq
        parts = last.unflatten(-1, (vq.partitions, vq.e_dim))
        return F.rms_norm(parts, (vq.e_dim,)).flatten(-2)


class NestedLatent(nn.Module):
    """LatentDragon's `nested` latent def around a vq: decoders read channel prefixes."""

    def __init__(self, dim: int, latent_chs: list, inner, unit: str = "channel", **_):
        super().__init__()
        name, config = _spec(inner)
        if name != "vq":
            raise ValueError(f"nested {name!r} is not supported here; use LatentDragon")
        self.inner = VQLatent(**{"dim": dim, **config})
        align = self.inner.vq.e_dim
        chs = sorted(latent_chs)
        if unit == "partition":
            chs = [ch * align for ch in sorted({1, *chs, dim // align})]
        if chs[-1] != dim:
            chs.append(dim)
        self.latent_chs = chs


def build_latent(spec) -> tuple[nn.Module, list[int]]:
    """The latent module and the channel counts a decoder can read."""
    name, config = _spec(spec)
    if name == "vq":
        return VQLatent(**config), [config["dim"]]
    if name == "nested":
        latent = NestedLatent(**config)
        return latent, latent.latent_chs
    raise ValueError(f"latent {name!r} is not supported here; use LatentDragon")


# ---------------------------------------------------------------- model


@dataclass
class LatentDragonEncoderOutput(ModelOutput):
    """`latents`: the encoder output before quantization `(B, C, h, w)`. `codes`: the
    quantized latent the decoder reads `(B, C, h, w)`. `indices`: local code indices per
    partition and level, `(B, h, w, P, R)`, each in `[0, codebook_size_per_partition)`.
    """

    latents: Tensor | None = None
    codes: Tensor | None = None
    indices: Tensor | None = None


@dataclass
class LatentDragonDecoderOutput(ModelOutput):
    """`sample`: the decoded image `(B, 3, H, W)` in about [-1, 1], not clamped."""

    sample: Tensor | None = None


class LatentDragonModel(PreTrainedModel):
    """A LatentDragon VQ autoencoder.

    `encode(x)` quantizes; `decode(codes)` or `decode_indices(indices)` reconstructs.
    `partitions=k` decodes from the first k partitions only (nested exports), the rest zeroed.
    """

    config_class = LatentDragonConfig
    base_model_prefix = "latentdragon"
    main_input_name = "pixel_values"

    def __init__(self, config: LatentDragonConfig):
        super().__init__(config)
        arch = config.arch
        domain, _ = _spec(arch.get("domain", "image"))
        if domain != "image":
            raise ValueError(
                f"domain {domain!r} is not supported here; use LatentDragon"
            )

        encoder, encoder_cfg = _spec(arch["encoder"])
        decoder, decoder_cfg = _spec(arch["decoder"])
        head, head_cfg = _spec(arch["head"])
        if (encoder, decoder, head) != ("latentmaid", "latentmaid", "conv"):
            raise ValueError(
                f"{encoder}/{decoder}/{head} is not supported here; use LatentDragon"
            )
        self.encoder = Encoder(**encoder_cfg)
        self.latent, self.latent_chs = build_latent(arch["latent"])
        self.decoder = Decoder(**decoder_cfg)
        self.head = Head(**head_cfg)
        self.post_init()

    def _init_weights(self, module):
        """Weights come from the checkpoint."""

    @property
    def vq_latent(self) -> VQLatent:
        """The vq latent def, inside a nested one or on its own."""
        return getattr(self.latent, "inner", self.latent)

    @property
    def quantizer(self) -> Quantizer:
        return self.vq_latent.vq

    def _mask(self, codes: Tensor, partitions: int | None) -> Tensor:
        """`codes (B, C, h, w)` with channels past the first `partitions` zeroed."""
        if partitions is None:
            return codes
        keep = partitions * self.quantizer.e_dim
        if keep not in self.latent_chs:
            sizes = [ch // self.quantizer.e_dim for ch in self.latent_chs]
            raise ValueError(f"partitions must be one of {sizes}, got {partitions}")
        mask = torch.zeros(codes.shape[1], dtype=codes.dtype, device=codes.device)
        mask[:keep] = 1
        return codes * mask.view(1, -1, 1, 1)

    def quantize(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        """Encoder output `(B, C, h, w)` -> codes `(B, C, h, w)` and indices `(B, h, w, P, R)`."""
        last = self.vq_latent.channels_last(latents)
        codes, indices = self.quantizer.quantize(last)
        return codes.movedim(-1, 1), indices

    def encode(self, pixel_values: Tensor, return_dict: bool = True):
        """Images `(B, 3, H, W)` in [-1, 1] -> latents, codes and indices."""
        latents = self.encoder(pixel_values)
        codes, indices = self.quantize(latents)
        out = LatentDragonEncoderOutput(latents=latents, codes=codes, indices=indices)
        return out if return_dict else out.to_tuple()

    def decode(
        self,
        codes: Tensor,
        partitions: int | None = None,
        force_not_quantize: bool = True,
        return_dict: bool = True,
    ):
        """Codes `(B, C, h, w)` -> images. `force_not_quantize=False` quantizes first, as
        for a diffusers `VQModel`; the default takes `codes` as already quantized."""
        if not force_not_quantize:
            codes, _ = self.quantize(codes)
        sample = self.head(self.decoder(self._mask(codes, partitions)))
        out = LatentDragonDecoderOutput(sample=sample)
        return out if return_dict else (sample,)

    def decode_indices(
        self, indices: Tensor, partitions: int | None = None, return_dict: bool = True
    ):
        """Indices `(B, h, w, P, R)` (or `(B, h, w, P)` for one level) -> images."""
        if indices.dim() == 4:
            indices = indices.unsqueeze(-1)
        codes = self.quantizer.lookup(indices).movedim(-1, 1)
        return self.decode(codes, partitions, return_dict=return_dict)

    def forward(
        self,
        pixel_values: Tensor,
        partitions: int | None = None,
        return_dict: bool = True,
    ):
        """Encode, quantize and decode: the reconstruction as `sample`."""
        encoded = self.encode(pixel_values)
        return self.decode(encoded.codes, partitions, return_dict=return_dict)
