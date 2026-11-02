# Precision

fp64 is the reference and plain PyTorch at the same input, weight and compute dtype is the
baseline. A row is marked `clear issue` at 10x the baseline error or a non-finite value, and
`large diff` at 2x. A mark sends a person to look at the row. Reports come from
`experiments/precision/kernels.py` and `experiments/precision/model_grads.py`.

## Kernels

Run on an RTX 5090, fp16 and bf16, three activations, up to 65536 codes x 256 channels.

| Op | Rows | clear issue | large diff |
| --- | --- | --- | --- |
| `quantize` | 36 | 0 | 0 |
| `mlp_project` (gemm) | 108 | 1 | 2 |

All three marked rows sit on `rel_max` at reference values near zero, with the same cosine as the
baseline to 10 digits, so they are not kernel errors. The older Triton `mlp_project` uses a tanh
approximation for gelu, and that shows as a real difference on gelu rows. No recipe uses gelu, and
`gemm` is exact.

## Model gradients

Trained weights of an f8 KL model (MLP encoder, SD-VAE decoder), real images, eval mode so fp64 and fp32 draw the same
noise. Convolutions run true fp32 here, with tf32 off.

| dtype | Encoder cos | Trunk cos | Head cos |
| --- | --- | --- | --- |
| fp32 | 1.000000 | 0.999989 | 0.999999 |
| bf16 | 0.999968 | 0.925557 | 0.971952 |

The kernels and compiled setup match the baseline at every dtype, so 0.93 is what bf16 rounding
does on these weights. Two bf16 runs on different code paths therefore cannot be compared by their
gradients; compare them at true fp32.

Weights trained in bf16 can overflow fp16: this model reaches activations near 1400, and a conv
sum overflows in the forward. Fresh runs do not have this problem. Use
`configs/ae/precision/bf16.py` to continue such weights.

## Equivalence with the research code

One training step at true fp32 against KohakuLatentMaid, taming, r3gan and stylegan2, with one and
two micro-batches, GAN gate open and closed: every group has cosine at least 0.99998 and norm
within 0.011%.

## Policies on real runs

Same seed and data, fresh weights, EMA loss at the end of a short run, no skipped steps in any run.

| Setup | Steps | bf16 | fp16 | bf16-true | fp16-true |
| --- | --- | --- | --- | --- | --- |
| f16 KohakuVQ, 65536 codes x 256 ch, conv decoder, critic idle | 3000 | 0.2369 | 0.2366 | 0.2371 | 0.2359 |
| f8 KL, MLP encoder, SD-VAE decoder | 1500 | 0.1846 | 0.1846 | | |
| SD-VAE, `configs/ae/recipes/sdvae.py` | 1500 | 0.1914 | 0.1903 | | |
| f16 KohakuVQ, 65536 codes x 256 ch, MLP decoder, MSE + LPIPS + DINOv3, taming critic from step 0 | 1500 | 0.3355 | 0.3306 | | |
| f8, MLP encoder, RMSNorm conv decoder, R3GAN critic from step 0 | 1500 | 0.1667 | 0.1650 | | |

fp16 ties or beats bf16 on all five, so it is the default. These are short runs. Stochastic
rounding in `adam16` is what makes the 16-bit weight policies work: with nearest rounding the same
runs ended at 0.2399 and 0.2381, because an update under half a step of the 16-bit grid is lost.

The DINOv3 loss runs here on random stand-in weights, because the real ones are gated, so its
timing is measured and its gradient precision is not.
