# Model

`Autoencoder = encoder + bottleneck + trunk + head`. `encode`, `bottleneck`, `decode` and
`head` are separately callable, `forward` composes them, and the model holds no loss
knowledge.

```python
from kohakuengine import load_config_file
from latentdragon.io.arch_json import arch_from_json
from latentdragon.model import build_autoencoder

arch = load_config_file("configs/ae/arch/sdvae.py").globals_dict["ARCH"]
model = build_autoencoder(arch_from_json(arch))
raw = model.encode(x)                # channels at dim 1
out = model.bottleneck(raw, offset)  # BottleneckOut
features = model.decode(out.z)       # trunk features
x_hat = model.head(features)
```

## Parts

| Registry | Entries |
|---|---|
| `ENCODER` | `mlp`, `conv`, `video` |
| `TRUNK` | `mlp`, `conv`, `video` |
| `BLOCK` | `mlp`, `res`, `mlp3d` |
| `NORM` | `rms`, `layer`, `group32` |
| `HEAD` | `regression` |

`norm_layer` takes a registry name, a dotted path or a factory. `build_block` is the only
place a block is built.

Stage tuples: `mlp` uses `(scale, hidden, mlp, blocks)`, `conv` uses
`(scale, hidden, blocks)`, `video` uses `(t_scale, scale, hidden, mlp, blocks)`. Order
matters. `mlp` and `video` trunks walk `block_configs` in reverse, so a config reads from
the pixel side. `conv` trunks walk it forward.

## One skeleton

`ShuffleEncoder` and `ShuffleTrunk` hold the pixel-shuffle stage walk for image and video.
The subclass picks the stage tuple, the block, the conv layer and the fuser at build; the
walk has no rank branch. `F.pixel_shuffle` accepts leading dimensions, so there is no
per-frame helper. A stage with no temporal scale gets `IdentityFuser`, which takes
`has_keyframe` like the real fusers, so the walk never tests a fuser's type.

The conv classes keep their own short walk because their stage owns its sampler and their
keys are `stages.N.blocks.M`.

## Attention placement

Each class keeps its historical slot, and the slots differ:

| Class | Where | Channels | Index |
|---|---|---|---|
| `mlp` encoder | after the stage's blocks | `hidden` | `block_configs` |
| `conv` encoder | after the sampler | `hidden` | `block_configs` |
| `mlp` trunk | after the stage proj, before the blocks | `hidden` | reversed internally, `flags[i]` matches `block_configs[i]` |
| `conv` trunk | before the stage | channels entering the stage | `block_configs` |

Video has no attention slots.

## Head

The old decoders split into trunk and `regression` head, so GAN schemes read `head.anchor`
rather than a decoder attribute, and later stage runs can swap the head. The trunk supplies
`head_form`:

| Trunk | Head |
|---|---|
| `mlp` | 1x1 conv |
| `conv` | norm, Mish, 3x3 conv |
| `video` | frame-wise 1x1 conv |

## Init

Each part runs `mup_init`; the head then runs `mup_init_output` on its projection. The
encoder output layer gets plain `mup_init` because the output variant gives a latent with a
very small std. `build_autoencoder` calls `bottleneck.init_encoder_out` after that, which is
how `kl` zeroes the logvar half, so a posterior starts at std near 1.

## Video

Pixels are `(B, T, C, H, W)`. Latents are `(B, C, T, H, W)`: the encoder and trunk swap
axes at the latent boundary so a bottleneck sees channels at dim 1 for any rank. The research
encoder returned `(B, T, C, H, W)` latents, so code written against it must transpose. The first
frame is a keyframe: it passes through the temporal conv and the fusers untouched, so a
clip has `1 + n * k` frames.

`forward_stream` runs one chunk with the cache from the previous chunk and returns the new
cache. Start with `has_keyframe=True` and an empty list, then pass `False`. The result
concatenates to the full forward. On the trunk it returns features; run the head on them.

Chunk sizes must be whole groups for the temporal scale of the encoder. The cache keeps the
last `kernel_size - 1` frames of history across chunks, even when a chunk is shorter than
that.

The temporal conv is a full `(C, C, 3)` Conv1d, not depthwise.

## Mechanics

`apply_recompute(model)` replaces `block.forward` with a non-reentrant checkpoint of the
original, so no block has a checkpoint branch and state dict keys stay as they are. Attention
blocks stay outside it, as before.

`COMPILE` sets the compile scope: `off`, `function` (norms and activations), `module` (blocks)
or `model` (encoder, trunk and head as one graph each). `compile_wrapper` marks a function or
block; `compile_parts` compiles whole parts. A wrapped function runs inline under an enclosing
compile, so the scopes do not conflict. A checkpointed block is compiled as one recompute
graph, and no scope adds an `_orig_mod` prefix to state dict keys.
