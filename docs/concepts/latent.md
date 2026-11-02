# Latent defs

A latent def is the serdes between the two halves of an autoencoder. It reads encoder slots and
writes the latent; it reads a latent, live or stored, and writes decoder slots. What both
directions share, such as a codebook, lives in it.

```python
encoded = latent_def.serialize(encoder_slots, offset=step)
encoded.latent            # what deserialize reads
encoded.losses            # live scalars, weighted by AUX_WEIGHTS name
encoded.logs              # detached scalars, keys carry their log section
slots = latent_def.deserialize(encoded.latent)
packed = latent_def.pack(encoded.latent)   # the storable form; deserialize reads it too
```

The single-slot defs read `input` (default `h`) and write `output` (default `z`), which is also
the slot the decoder gets.

| Name | Latent slot | Loss | Behaviour |
|---|---|---|---|
| `identity` | the encoder slot | none | passes the encoder output |
| `kl` | the sample | `kl` | Gaussian posterior over half of the encoder channels each |
| `vq` | `{"codes", "indices"}` | `vq` | quantizes with a `QUANTIZER` spec; `pack` keeps the indices |
| `nested` | the inner def's, plus a mask | the inner's | per-sample channel masks of the decoder input |
| `slots` | every def's own | prefixed by def name | independent defs side by side |

## kl

The encoder emits mean and logvar along dim 1. Logvar clamps to `logvar_clamp`, default
`(-30, 20)`. Training samples `mean + exp(logvar / 2) * noise`; eval uses the mean. The KL
against a standard normal is summed over every non-batch dim, then meaned over the batch.
`init_from_encoder` zeroes the logvar half of the encoder's last layer, so the posterior starts
with unit std and a KL near zero. `raw` is the mean.

The KL term has weight 0 until `AUX_WEIGHTS` sets `kl`, for example `{"kl": 1e-6}`.

## vq

```python
{"name": "vq", "dim": 256, "quantizer": {"name": "kohaku", "n_embed": 16384}}
```

`dim` is the slot's channel count. While training and evaluating, the latent slot holds the
straight-through codes and the global indices; `deserialize` reads the codes. `pack` keeps only
the indices, and `deserialize` of a packed latent looks them up in the same codebook, so the
stored latent is what crosses the wire and the codebook is the reader. Indices are
channels-last in the latent's own order: `(B, H, W, P)` for an image latent with `R = 1`. Its
logs are `vq/commitment_loss`, `vq/codebook_loss`, `vq/usage`, `vq/perplexity` and
`vq/perplexity_rate`; the task keeps the `vq/usage_window` code-usage window.

## nested

`nested` trains a latent whose channel prefixes each decode on their own.

```python
{"name": "nested", "dim": 64, "inner": "kl", "latent_chs": [8, 16, 32, 64]}
```

`latent_chs` is sorted, and `dim` is appended when missing. While training, sample `i` keeps the
channels of `masks[(offset + i) % count]`; the task passes the step as `offset` so every sample
sees every mask over time. `ratio` repeats masks by weight. The mask travels in the latent as
`<output>_mask` and applies to the decoder input, so the KL and the VQ loss cover the full latent.

Eval keeps the latent whole. `eval_variants()` lists the channel counts, and validation and
sample grids decode `variant(latent, count)` for each.

Around a `vq` inner, `unit="partition"` counts partitions; one partition and all partitions are
always included. Channel widths must be multiples of the partition width. `jitter` shifts each
sample's width index by a uniform draw from {-1, 0, +1}, clamped to the available widths.

## slots

```python
{"name": "slots", "defs": {
    "key": {"name": "kl", "input": "h_key", "output": "key"},
    "motion": {"name": "vq", "dim": 8, "quantizer": {...}, "input": "h_motion", "output": "motion"},
}}
```

Each def keeps its own slot names; losses and logs are prefixed with the def's name (`key/kl`,
`motion/vq`). `examples/packed_latent/` trains this and decodes a stored packed latent.
