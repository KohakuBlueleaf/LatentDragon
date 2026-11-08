# Latent slots

Two kinds of latent work touch training differently.

| Slot | Contract | Fits |
| --- | --- | --- |
| `LATENT_VIEW` | `(x, latent, step) -> (x', latent')`, extra decode and recon | equivariance, masking |
| `LATENT_TERM` | `(LatentCtx) -> (loss, logs)`, no decode | alignment, regularizers |
| `TEACHER` | frozen `features(x)` of the data slots, never checkpointed | feature models |
| `WEIGHTING` | `fixed`, `scheduled`, `adaptive` | term weights |

A view declares `adversarial` (the critic sees it), `codebook_safe` (it stays on the
codebook) and `domains`. Each view has its own recon pass, so views may change resolution. A
term declares `requires`, `teacher` and `group`. Owned parameters train and checkpoint but
never reach exports. A term that runs its own passes (the `mae` term's masked encoder walk)
gets the uncompiled model once in `bind(model)`, before it moves to the device, and keeps the
parts it calls as plain attributes so their parameters stay the model's; `post_step(step, sync)`
runs once per optimizer step after the weight EMA, for term state shared across ranks.

Terms and latent def losses share one path: live scalars in the composed backward, reaching
the encoder without passing through `x_hat`. Seed logic is unchanged by them.

Config checks refuse work that cannot act: encoder-side terms with a frozen encoder, views
with decoder and head frozen, and non-codebook-safe views on a latent def with a codebook unless
`ALLOW_OFF_CODEBOOK`. `LatentCtx` carries the data slots, the `Encoded` record (its latent def's
`raw` view is the mean of a `kl`), the reconstruction with its seeds, and `states`: the main
pass's slot-boundary values by the part that wrote them, `encoder` (before the latent def),
`latent` and `decoder` (the output latent the head reads). A term reads any of them, and
`parts_of(spec)` names the parts it trains for the freeze check.

## State RMS

`state_rms` pulls the per-position RMS of a state toward `target`, as a loss instead of a norm.
Its knobs: `at` (`encoder`, `latent`, `decoder`), `slot` (one slot path, or every float
tensor), `target`, `groups` (channel groups, each with its own RMS) and `channel_dim`.

```python
LATENT_TERMS = [
    # Output latent: holds the decoder's scale against the head's
    {"term": {"name": "state_rms", "at": "decoder"},
     "weight": {"name": "fixed", "weight": 0.01}},
    # Encoder output before a partition-wise RMS-normed vq: groups = partitions
    {"term": {"name": "state_rms", "at": "encoder", "groups": 16},
     "weight": {"name": "fixed", "weight": 0.01}},
]
```

Its logs are `train/state_rms/<at>/<slot>_loss` and `_rms`; `encoder_output/` and
`output_latent/` log the mean, std and largest magnitude of both states on metrics steps.

No EQ-VAE or VA-VAE ships. They are expected to arrive as one view and one term.
