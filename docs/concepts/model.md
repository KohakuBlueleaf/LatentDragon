# Model

An autoencoder is three arrows over named slots:

```text
domain -> encoder -> latent.serialize -> latent -> latent.deserialize -> decoder -> head -> domain
```

Every value crossing an arrow is a dict of named slots. A slot holds a batch-first tensor or a
pytree of them; that is the only structural rule. The domain names the data slots, each part
declares the slots it reads (`inputs`) and writes (`outputs`), and the build refuses a part that
reads a slot the one before it does not write. Shapes, widths, counts and dtypes inside a slot
belong to the parts.

```python
from kohakuengine import load_config_file
from latentdragon.model import build_autoencoder

arch = load_config_file("configs/ae/arch/sdvae.py").globals_dict["ARCH"]
model = build_autoencoder(arch)

encoded = model.encode({"x": x})          # Encoded: latent, losses, logs, extras
features = model.decode(encoded.latent)   # the output latent, e.g. {"y": ...}
x_hat = model.head(features)              # data slots, e.g. {"x": ...}
packed = model.latent.pack(encoded.latent)
x_hat = model.head(model.decode(packed))  # a stored latent decodes the same way
```

## ARCH

```python
ARCH = {
    "domain": {"name": "image"},
    "encoder": {"name": "sdvae", "out_channels": 8, "stages": [...], "norm": "group32"},
    "latent": "kl",
    "decoder": {"name": "sdvae", "in_channels": 4, "stages": [...], "norm": "group32"},
    "head": {"name": "conv", "in_channels": 128, "kernel_size": 3, "norm": "group32", "act": "mish"},
}
```

Each entry is a registry choice: a name, a dotted path, or `{"name": ..., **kwargs}`. The
framework passes no arguments of its own, so every width and slot name is written in the
entry. `domain` defaults to `image` and `latent` to `identity`.

## The contract

`latentdragon/model/core/contracts.py` defines it.

| Part | Registry | Call | Declares |
| --- | --- | --- | --- |
| encoder | `ENCODER` | `forward(slots) -> slots` | `inputs`, `outputs` |
| latent def | `LATENT` | `serialize(slots, offset) -> Encoded`, `deserialize(latent) -> slots`, `pack(latent) -> latent` | `inputs`, `outputs`, `decoder_slots` |
| decoder | `DECODER` | `forward(slots) -> slots` | `inputs`, `outputs` |
| head | `HEAD` | `forward(slots) -> slots` | `inputs`, `outputs`, `objective` |

`Encoded` carries `latent` (what `deserialize` reads), `losses` (live scalars the task weights
by name through `AUX_WEIGHTS`), `logs` (detached scalars whose keys carry their section) and
`extras` (private to the latent def). See [latent](latent.md) for the latent defs.

`TensorPart` is a convenience base for a part with one input slot and one output slot: it
implements `forward` around a slot-local `tensor(x)`.

## Capabilities

Anything else a part offers is optional, and whoever needs it looks it up and refuses clearly
when it is missing.

| Capability | On | Used by |
| --- | --- | --- |
| `last_layer` | head | adaptive GAN weight (`taming` with `adaptive_weight`) |
| `last_layer` | encoder | latent-term `adaptive` weighting; a latent def's `init_from_encoder` |
| `spatial_scale`, `temporal_scale` | encoder, decoder | the scale check at build; tiled decoding |
| `reach`, `scales`, `forward_early`, `forward_late`, `forward_stream`, `tensor` | decoder | decode strategies, see [inference](inference.md) |
| `forward_stream` | encoder | `stream_encode` |
| `raw`, `eval_variants`, `variant` | latent def | latent stats, sample previews, nested evaluation |
| `codebook_size`, `codes`, `quantizer` | latent def | code usage in training and validation |
| `objective`, `needs_pixels` | head | objective selection and config checks |
| `function_scope_gaps` | decoder | a compile-scope note |

## Built-in families

| Registry | Name | Slots | Stage tuples |
| --- | --- | --- | --- |
| `ENCODER` | `latentmaid` | `x -> h` | `(scale, hidden, mlp, blocks)` |
| `ENCODER` | `latentmaid_video` | `x -> h` | `(t_scale, scale, hidden, mlp, blocks)` |
| `ENCODER` | `sdvae` | `x -> h` | `(scale, hidden, blocks)` |
| `DECODER` | `latentmaid` | `z -> y` | `(scale, hidden, mlp, blocks)`, the last runs first |
| `DECODER` | `latentmaid_video` | `z -> y` | as its encoder, the last runs first |
| `DECODER` | `sdvae` | `z -> y` | `(scale, hidden, blocks)`, the first runs first |
| `HEAD` | `conv` | `y -> x` | norm, activation, `kernel_size` projection; `video` projects per frame |
| `HEAD` | `identity` | renames slots | for decoders that end in the signal |

Every slot name is a keyword (`input`, `output`), so the same family serves any slot. The
blocks, norms and stage helpers the families share live in `latentdragon/model/nn/`.

`latentmaid` stages are pixel-unshuffle (encoder) or pixel-shuffle (decoder) walks of MLP
blocks; `sdvae` stages are residual conv stages with stride-2 down and nearest-neighbour up.
Attention slots differ per family:

| Family | Where | Channels |
| --- | --- | --- |
| `latentmaid` encoder | after the stage's blocks | `hidden` |
| `sdvae` encoder | after the sampler | `hidden` |
| `latentmaid` decoder | after the stage proj, before the blocks; `attn[i]` matches `stages[i]` | `hidden` |
| `sdvae` decoder | before the stage | channels entering the stage |

The video families have no attention slots.

## Unit-row blocks

The `mlp` and `res` blocks take `unit_rows`, set through the block spec:
`"block": {"name": "mlp", "unit_rows": True}`. Every linear and conv inside the block then
applies its weight with unit-norm rows, so the raw weight's scale never reaches an activation:
Adam's growth of a raw weight only shrinks its later steps, and the block's hidden states and
writes stay bounded. The block's norms drop their affine, and its writer (`swiglu.out`, or
`conv2` in `res`) carries a per-channel gain, the only scale the block owns. Biases stay.
Layers outside the blocks (stage projections, attention slots, the encoder's output projection,
the head) are unchanged.

`writer_init="output"` scales a unit-row writer's gain by `fan_in ** -0.5` in place of its
weight, which gives the same initial write. The rows are normalized at use, so EMA weights and
loaded checkpoints need no extra step.

## Init

Each part runs `mup_init`; the `conv` head then runs `mup_init_output` on its projection. When
the encoder declares `last_layer` and the latent def has `init_from_encoder`, the build calls it:
`kl` zeroes the logvar half, so a posterior starts at std near 1.

## Video

Clips are `(B, T, C, H, W)`. The LatentMaid video encoder writes `(B, C, T, H, W)` and its
decoder reads that and writes `(B, T, C, H, W)` features. The first frame is a keyframe: it passes
through the temporal conv and the fusers untouched, so a clip has `1 + n * k` frames.

`forward_stream` runs one chunk with the cache from the previous chunk and returns the new
cache. Start with `has_keyframe=True` and an empty list, then pass `False`. The result
concatenates to the full forward. Chunk sizes must be whole groups for the encoder's temporal
scale. The temporal conv is a full `(C, C, 3)` Conv1d, not depthwise.

## Mechanics

`apply_recompute(model)` replaces `block.forward` with a non-reentrant checkpoint of the
original, so no block has a checkpoint branch and state dict keys stay as they are.

`COMPILE` sets the compile scope: `off`, `function` (norms and activations), `module` (blocks)
or `model` (encoder, decoder and head as one graph each). The latent def stays eager, because
its kernels launch Triton. A wrapped function runs inline under an enclosing compile, so the
scopes do not conflict, and no scope adds an `_orig_mod` prefix to state dict keys.

## Examples

`examples/` holds two runnable toys whose parts are all registered from outside the framework:
paired video and audio data with one unified latent, and a clip packed into a keyframe latent
and a vector-quantized motion latent. See [examples](../../examples/README.md).
