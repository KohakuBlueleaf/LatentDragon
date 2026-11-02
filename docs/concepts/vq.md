# Vector quantization

KohakuVQ generates its codebook from a small trainable base with a shared MLP, so
`c_k = s * f(b_k)` with `b_k` of dimension 8 to 16 (see the README). The recorded configuration is a
single codebook: 65536 codes, `partitions=1`, `rvq_levels=1`, base 16, hidden 2048.

`latentdragon.vq` has one quantizer class, `GridQuantizer`, over a partition x residual grid.
A latent of `C` channels splits into `P` partitions of `D = C / P` channels, and each
partition descends `R` residual levels, each level with `N` codes.

```python
from latentdragon.vq import QUANTIZER

vq = QUANTIZER.get("kohaku")(
    emb_dim=256, n_embed=65536, partition_mode="hyper",
    mlp_projector=True, trainable_base=True, dim_scale=True, base_dim=16,
    ivq_loss_weight=1e-3,
)
out, info = vq(z)                     # z and out are channels-last (..., C)
z_q = vq.get_codebook_entry(info["indices"])
```

## Quantizers

| Name | Codebook source | Notes |
|---|---|---|
| `kohaku` | `projected` | KVQ: codes are a projection of a base table |
| `naive` | `direct` | classic VQ table, uniform init in `(-1/N, 1/N)` |

They are both a `GridQuantizer`, so `partitions` and `rvq_levels` mean the same thing in
each. Backward is the straight-through estimator only.

## Codebook sources

A codebook is a module whose `forward()` returns `(R, P, N, D)`.

`projected` computes `scale * proj(base)`:

- `mlp_projector=False`: `base @ w + b`.
- `mlp_projector=True`: `w2 @ act(w1 @ base + b1) + b2`, `mlp_act` is `silu`, `mish` or
  `gelu`, `mlp_hidden_dim` sets the hidden width.
- `trainable_base=False` keeps the base a buffer; `True` makes it a parameter. The state
  dict key is `codebook_base` either way.
- `base_dim` lets the base be smaller than `D`.
- `codebook_scale` holds one scalar per level. `dim_scale=True` adds
  `codebook_dim_scale`, one scalar per level and dimension.

Init aims at an output std near `D^-1/2`. A linear projector has unit gain, so the base
uses `D^-1/2`. An MLP projector with SiLU scales the output variance by about 1/4 (std by 1/2), so the base uses `2 D^-1/2`.
Projector weights use `fan_in^-1/2` and biases start at zero.

Because every code passes through shared projector weights, an update from any used code
moves all codes. This is the design reason a projected codebook is expected to resist
collapse; there are usage and perplexity numbers at 65536 codes but no ablation
against a plain table.

## Sharing

`partition_mode` picks how partitions relate to tables and projector groups `G`.

| Mode | Base | Projector | Indices |
|---|---|---|---|
| `shared` | `(N, b)` | `G = R`, output `D` | local to one table |
| `indep` | `(P, N, b)` | `G = P R`, output `D` | offset by partition |
| `hyper`, `separated_proj=True` | `(N, b)` | `G = P R`, output `D` | offset by partition |
| `hyper`, `separated_proj=False` | `(N, b)` | `G = R`, output `P D` | offset by partition |

`direct` supports `shared` and `indep`. `hyper` needs a projector, so it raises there.

## Indices

Indices are global. For level `r`, partition `p` and local code `i`:

```text
index = i + p * N * non_shared + r * codes_per_level
codes_per_level = N * (P if non_shared else 1)
codebook_size = codes_per_level * R
```

The shape is `(..., P)` for `R = 1` and `(..., P, R)` otherwise. `get_codebook_entry`
removes the offsets and sums the levels.

## Losses

`vq_loss = beta * commitment + codebook + ivq_loss_weight * ivq`

- `commitment` and `codebook` compare the summed `z_q` to the input.
- `ivq` runs per level: each code moves to its nearest detached point of that level's
  residual, then the levels are meaned. A code nothing selects still gets a gradient,
  so dead codes drift back to the data. It is a regularizer, and on a degenerate `naive`
  init it can pull the whole table to the data centroid, so keep the weight small.

## Metrics

`info` carries `usage`, `perplexity`, `perplexity_rate`, `commitment_loss`,
`codebook_loss`, `ivq_loss` when enabled, and `perplexity_rate_prod` for `P > 1`.
Stats stay in fp32 for any input dtype, since entropy over a large codebook loses too
much in fp16.

These are per micro-batch values, so `usage` is bounded by the number of tokens in the batch:
8192 tokens give at most 0.125 of a 65536-code book, even when every token picks a different
code. Validation reports `val/vq_total_usage` and `val/vq_total_perplexity_rate` over the whole
validation set; read those for codebook health.

| Grid | `usage` | `perplexity_rate` |
|---|---|---|
| `P = 1` | used / `codebook_size` | perplexity / `codebook_size` |
| `P > 1`, `R = 1` | geometric mean over partitions | mean per-partition rate |
| `P > 1`, `R > 1` | pooled over global indices | mean per-partition rate, meaned over levels |

The geometric mean reflects the product space of a partitioned code. The pooled form in
the last row keeps metric continuity with earlier runs.

## Kernels

A quantizer starts on eager kernels. `bind_kernels(device)` resolves `quantize`, `nearest`
and `mlp_project` once, so `forward` holds no device branch.
