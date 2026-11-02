"""Op names and their signatures, implemented by every backend.

quantize(z[..., parts, dim], codebook[parts, codes, dim])
    -> (z_q[..., parts, dim], z_q_diff[..., parts, dim], indices[..., parts],
        min_dists[..., parts])
    nearest code per partition by squared L2. `z_q` and `z_q_diff` keep `z.dtype`,
    `min_dists` is at least fp32. Backward: `z_q` gives no gradient to `z` and
    scatters onto the selected codes; `z_q_diff = z - z_q` is identity to `z` and
    scatters negated onto the selected codes. No gradient for `min_dists`.

nearest(z[..., parts, dim], codebook[parts, codes, dim])
    -> (nearest_z[..., parts, dim], indices[..., parts], dists[..., parts])
    the same search and gather, without gradients.

mlp_project(base[groups, rows, in_dim], w1[groups, in_dim, hidden],
            b1[groups, 1, hidden], w2[groups, hidden, out_dim],
            b2[groups, 1, out_dim], act="silu", trainable_base=True)
    -> out[groups, rows, out_dim]
    w2 @ act(w1 @ base + b1) + b2 computed in at least fp32. `act` is one of
    `ACTIVATIONS`. `trainable_base=False`: `base` gets no gradient. The Triton gelu is the
    tanh approximation and the eager one is exact, so the two backends differ on gelu.

adam16(param, grad, exp_avg, exp_rms, bad, step, lr, betas, eps, weight_decay,
       stochastic=False, seed=0) -> None
    one in-place AdamW step in fp32 registers, state in its own dtype, `exp_rms` holding
    the root of the second moment. Nothing moves when the device flag `bad` is set.
    `stochastic` rounds 16-bit writebacks by distance.
"""

QUANTIZE = "quantize"
NEAREST = "nearest"
MLP_PROJECT = "mlp_project"
ADAM16 = "adam16"
OPS = (QUANTIZE, NEAREST, MLP_PROJECT, ADAM16)
ACTIVATIONS = ("silu", "gelu", "mish")
