# Latent slots

Two kinds of latent work touch training differently.

| Slot | Contract | Fits |
| --- | --- | --- |
| `LATENT_VIEW` | `(x, z, step) -> (x', z')`, extra decode and recon | equivariance, masking |
| `LATENT_TERM` | `(LatentCtx) -> (loss, logs)`, no decode | alignment, regularizers |
| `TEACHER` | frozen `features(x)`, never checkpointed | feature models |
| `WEIGHTING` | `fixed`, `scheduled`, `adaptive` | term weights |

A view declares `adversarial` (the critic sees it), `codebook_safe` (it stays on the
codebook) and `domains`. Each view has its own recon pass, so views may change resolution. A
term declares `requires`, `teacher` and `group`. Owned parameters train and checkpoint but
never reach exports.

Terms and bottleneck losses share one path: live scalars in the composed backward, reaching
the encoder without passing through `x_hat`. Seed logic is unchanged by them.

Config checks refuse work that cannot act: encoder-side terms with a frozen encoder, views
with trunk and head frozen, and non-codebook-safe views on VQ unless `ALLOW_OFF_CODEBOOK`.
`LatentCtx.z_raw` is the bottleneck's `raw`, the mean on a KL bottleneck.

No EQ-VAE or VA-VAE ships. They are expected to arrive as one view and one term.
