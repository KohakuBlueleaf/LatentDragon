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
never reach exports.

Terms and latent def losses share one path: live scalars in the composed backward, reaching
the encoder without passing through `x_hat`. Seed logic is unchanged by them.

Config checks refuse work that cannot act: encoder-side terms with a frozen encoder, views
with decoder and head frozen, and non-codebook-safe views on a latent def with a codebook unless
`ALLOW_OFF_CODEBOOK`. `LatentCtx` carries the data slots, the `Encoded` record (its latent def's
`raw` view is the mean of a `kl`), and the reconstruction with its seeds.

No EQ-VAE or VA-VAE ships. They are expected to arrive as one view and one term.
