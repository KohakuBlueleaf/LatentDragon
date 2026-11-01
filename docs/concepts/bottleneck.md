# Bottlenecks

A bottleneck sits between encoder and trunk. It takes channels at dim 1 for any rank, so
the same code serves images, video and other signals. Nothing here assumes a domain.

```python
out = bottleneck(raw, offset=step)
out.z        # latent for the trunk
out.aux      # live scalars for the composed backward
out.metrics  # detached scalars for logging
out.indices  # code indices or None
out.raw      # latent before quantization or sampling
```

Build one with `build(spec, BOTTLENECK, latent_dim=C)`.

| Name | `in_channels(C)` | aux | Behaviour |
|---|---|---|---|
| `identity` | `C` | none | passes the encoder output |
| `kl` | `2 C` | `kl` | Gaussian posterior |
| `vq` | `C` | `vq` | runs a `QUANTIZER` spec, see `vq.md` |
| `nested` | from `inner` | from `inner` | per-sample channel masks |

## kl

The encoder emits mean and logvar along dim 1. Logvar clamps to `logvar_clamp`, default
`(-30, 20)`. Training samples `mean + exp(logvar / 2) * noise`; eval uses the mean. The
KL against a standard normal is summed over every non-batch dim, then meaned over the
batch. `init_encoder_out(layer, C)` zeroes the logvar half of the output layer, so the
posterior starts with unit std and a KL near zero. `raw` is the mean.

The KL term has weight 0 until `AUX_WEIGHTS` sets `kl`, for example `{"kl": 1e-6}`.

## vq

The spec names a quantizer, and `emb_dim` comes from `latent_dim`:

```python
{"name": "vq", "quantizer": {"name": "kohaku", "n_embed": 16384}}
```

`indices` are channels-last in the latent's own order: `(B, H, W, P)` for an image latent
with `R = 1`. Usage tracking windows belong to the task, not the quantizer.

## nested

`nested` trains a latent whose channel prefixes each decode on their own.

```python
{"name": "nested", "inner": "kl", "latent_chs": [8, 16, 32, 64]}
```

`latent_chs` is sorted, and `latent_dim` is appended when missing. While training, sample
`i` keeps the channels of `masks[(offset + i) % count]`; the task passes the step as
`offset` so that every sample sees every mask over time. `ratio` repeats masks by weight.

Eval returns the latent unmasked. `eval_masks()` lists the channel counts, and an
evaluator zeroes `z[:, ch:]` for each one.

The mask applies to the sampled latent, and the KL still covers every channel.

Around a `vq` inner the mask applies after quantization, so the indices, the usage metrics and
the VQ loss are those of the full latent, and the decoder sees only the kept partitions:

```python
{"name": "nested", "unit": "partition", "latent_chs": [2, 3], "jitter": 1,
 "inner": {"name": "vq", "quantizer": {"name": "kohaku", "n_embed": 16, "partitions": 4}}}
```

With `unit="partition"` the widths count partitions; one partition and all partitions are always
included. Channel widths must be multiples of the partition width. `jitter` shifts each sample's
width index by a uniform draw from {-1, 0, +1}, clamped to the available widths. Validation names
carry channel counts.
