# Performance

Measured on RTX 5090 32 GB cards, PCIe 4 x16, torch 2.15. Throughput is images per second from the
median step. The DINOv3 loss runs on random stand-in weights, which does not change its timing.
`experiments/throughput/recipes.py` produces these rows; precision is in
[precision.md](precision.md).

## Four GPUs, real ImageNet JPEGs

| Recipe | Before | After | Gain |
| --- | --- | --- | --- |
| `kvq_f16c256_production` | 270.8 | 827.1 | 3.05x |
| `production_active` | 257.9 | 781.2 | 3.03x |
| `kvq_f16c256` | 154.4 | 387.2 | 2.51x |
| `taming_active` | 148.5 | 359.2 | 2.42x |
| `latentmaid_0222_f8c4` | 137.1 | 233.9 | 1.71x |
| `latentmaid_f8c4` | 110.2 | 206.5 | 1.87x |
| `r3gan_active` | 103.5 | 186.1 | 1.80x |
| `sdvae_f8c4` | 79.3 | 129.5 | 1.63x |

All rows are fp16 on a bf16 wire except `r3gan_active`, which ran in bf16; fp16 and bf16 run at the
same speed on the same wire. Data does not limit these rates: 839 img/s on real data against 845
synthetic on production, and 60.8 against 60.3 on one GPU of `latentmaid_0222_f8c4`.

## One GPU, synthetic data

| Recipe | Before | Checkpoints | No checkpoints | Gain |
| --- | --- | --- | --- | --- |
| `mse_only` | 88.4 | 244.0 | 351.8 | 3.98x |
| `kvq_f16c256_production` | 70.3 | 175.9 | 224.0 | 3.19x |
| `production_active` | 66.8 | 163.1 | 204.4 | 3.06x |
| `taming_acc` | 24.8 | 69.1 | 75.6 | 3.04x |
| `video` (clips/s) | 9.5 | 21.2 | 25.4 | 2.67x |
| `kvq_f16c256_stylegan2` | 39.6 | 90.0 | 103.3 | 2.61x |
| `kvq_f16c256` | 39.8 | 89.0 | 102.8 | 2.58x |
| `taming_active` | 38.4 | 82.6 | 93.4 | 2.44x |
| `latentmaid_f8c4_r3gan` | 26.9 | 53.0 | 62.1 | 2.31x |
| `kvq_grid` | 47.6 | 94.9 | 109.8 | 2.30x |
| `latentmaid_0222_f16c64n16` | 47.5 | 95.0 | 109.2 | 2.30x |
| `naive_vq` | 48.0 | 95.2 | 109.8 | 2.29x |
| `kvq_nested` | 48.2 | 94.4 | 109.5 | 2.28x |
| `latentmaid_f8c4` | 28.4 | 52.4 | 61.0 | 2.15x |
| `latentmaid_0222_better_dec_f8c4` | 34.8 | 62.5 | 72.6 | 2.09x |
| `latentmaid_0222_f8c4` | 36.2 | 59.8 | 68.7 | 1.90x |
| `r3gan_active` | 25.0 | 47.0 | out of memory | 1.88x |
| `latentmaid_f8c4_stylegan2` | 28.5 | 52.6 | out of memory | 1.85x |
| `stylegan2_active` | 27.3 | 48.8 | out of memory | 1.79x |
| `sdvae_f8c4` | 21.0 | 33.3 | 37.2 | 1.77x |

Recipes with checkpoints on can run without them where memory allows; "No checkpoints" uses up to
31 GB on the f8 recipes.

## What the setup is

| Knob | Value | Measured |
| --- | --- | --- |
| `COMPILE` | `model` | 1.3x to 1.5x on the production step; other scopes are for debugging |
| `COMPILE_LOSS` | on | LPIPS 60.6 to 30.6 ms, DINOv3 30.5 to 21.2 ms |
| `GRAD_CKPT` | off for KVQ, on for f8 | 12 to 27% faster off, at twice the memory |
| `CHANNELS_LAST` | on for sdvae and 0222 | +5 to 10% there, nothing on KVQ |
| `CUDNN_BENCHMARK` | on for sdvae and 0222 | +5% |
| `REDUCER` | bucketed, 8 MiB | 32.7 to 4.6 ms of reduce per step against one flat all-reduce |
| `REDUCE_DTYPE` | `bf16` | 845 against 787 img/s with an fp32 wire |
| `CPU_THREADS` | 2 per rank | 338 to 677 img/s on four GPUs |
| `PRECISION` | `fp16` | same speed as bf16; see precision |
| `COMPILE_MODE` | `default` | `reduce-overhead` with the flat reducer: 222 against 224 img/s |
