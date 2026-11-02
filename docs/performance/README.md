# Performance

Measured on RTX 5090 32 GB cards, PCIe 4 x16, torch 2.15. Throughput is images per second from the
median step, at 256 px. Each row names the setup it ran; "critic idle" means the GAN had not started. The DINOv3 loss runs on random stand-in weights, which does not change its timing.
`experiments/throughput/recipes.py` produces these rows; precision is in
[precision.md](precision.md).

## Four GPUs, real ImageNet JPEGs

| Setup | Before | After | Gain |
| --- | --- | --- | --- |
| f16 KohakuVQ, 65536 codes x 256 ch, MLP decoder, MSE + LPIPS + DINOv3, critic idle | 270.8 | 827.1 | 3.05x |
| the same, taming critic from step 0 | 257.9 | 781.2 | 3.03x |
| f16 KohakuVQ, 65536 codes x 256 ch, conv decoder, MSE + LPIPS, critic idle | 154.4 | 387.2 | 2.51x |
| the same, taming critic from step 0 | 148.5 | 359.2 | 2.42x |
| f8 KL, MLP encoder, SD-VAE decoder | 137.1 | 233.9 | 1.71x |
| f8, MLP encoder, RMSNorm conv decoder, no KL | 110.2 | 206.5 | 1.87x |
| the same, R3GAN critic from step 0 | 103.5 | 186.1 | 1.80x |
| SD-VAE, `configs/ae/recipes/sdvae.py` | 79.3 | 129.5 | 1.63x |

All rows are fp16 on a bf16 wire except the R3GAN row, which ran in bf16; fp16 and bf16 run at the
same speed on the same wire. Data does not limit these rates: 839 img/s on real data against 845
synthetic on the first row, and 60.8 against 60.3 on one GPU of the f8 KL MLP-encoder row.

## One GPU, synthetic data

| Setup | Before | Checkpoints | No checkpoints | Gain |
| --- | --- | --- | --- | --- |
| f16 KohakuVQ, 65536 codes x 256 ch, MLP decoder, MSE only | 88.4 | 244.0 | 351.8 | 3.98x |
| f16 KohakuVQ, 65536 codes x 256 ch, MLP decoder, MSE + LPIPS + DINOv3, critic idle | 70.3 | 175.9 | 224.0 | 3.19x |
| the same, taming critic from step 0 | 66.8 | 163.1 | 204.4 | 3.06x |
| f16 KohakuVQ, 65536 codes x 256 ch, conv decoder, taming critic from step 0, 4 micro-batches | 24.8 | 69.1 | 75.6 | 3.04x |
| video, f16 x 6 frames, 32 ch, 13-frame clips (clips/s) | 9.5 | 21.2 | 25.4 | 2.67x |
| f16 KohakuVQ, 65536 codes x 256 ch, RMSNorm conv decoder, StyleGAN2 critic idle | 39.6 | 90.0 | 103.3 | 2.61x |
| f16 KohakuVQ, 65536 codes x 256 ch, conv decoder, MSE + LPIPS, critic idle | 39.8 | 89.0 | 102.8 | 2.58x |
| the same, taming critic from step 0 | 38.4 | 82.6 | 93.4 | 2.44x |
| f8, MLP encoder, RMSNorm conv decoder, R3GAN critic idle | 26.9 | 53.0 | 62.1 | 2.31x |
| f16 KohakuVQ, 4 partitions x 2 residual levels of 4096 codes | 47.6 | 94.9 | 109.8 | 2.30x |
| f16 nested KL, 64 ch cycled over 8/16/32/64 | 47.5 | 95.0 | 109.2 | 2.30x |
| f16 plain codebook, 16384 codes, 2 residual levels | 48.0 | 95.2 | 109.8 | 2.29x |
| f16 KohakuVQ, 8 partitions of 4096 codes, nested over partitions | 48.2 | 94.4 | 109.5 | 2.28x |
| f8, MLP encoder, RMSNorm conv decoder, no KL | 28.4 | 52.4 | 61.0 | 2.15x |
| f8 KL, MLP encoder, RMSNorm conv decoder | 34.8 | 62.5 | 72.6 | 2.09x |
| f8 KL, MLP encoder, SD-VAE decoder | 36.2 | 59.8 | 68.7 | 1.90x |
| f8, MLP encoder, RMSNorm conv decoder, R3GAN critic from step 0 | 25.0 | 47.0 | out of memory | 1.88x |
| f8, MLP encoder, RMSNorm conv decoder, StyleGAN2 critic idle | 28.5 | 52.6 | out of memory | 1.85x |
| the same, StyleGAN2 critic from step 0 | 27.3 | 48.8 | out of memory | 1.79x |
| SD-VAE, `configs/ae/recipes/sdvae.py` | 21.0 | 33.3 | 37.2 | 1.77x |

Setups with gradient checkpointing on can run without it where memory allows; "No checkpoints"
uses up to 31 GB on the f8 setups.

## What the setup is

| Knob | Value | Measured |
| --- | --- | --- |
| `COMPILE` | `model` | 1.3x to 1.5x on the f16 KohakuVQ MLP-decoder step; other scopes are for debugging |
| `COMPILE_LOSS` | on | LPIPS 60.6 to 30.6 ms, DINOv3 30.5 to 21.2 ms |
| `GRAD_CKPT` | off for KVQ, on for f8 | 12 to 27% faster off, at twice the memory |
| `CHANNELS_LAST` | on for the SD-VAE decoder setups | +5 to 10% there, nothing on KohakuVQ |
| `CUDNN_BENCHMARK` | on for the SD-VAE decoder setups | +5% |
| `REDUCER` | bucketed, 8 MiB | 32.7 to 4.6 ms of reduce per step against one flat all-reduce |
| `REDUCE_DTYPE` | `fp16` for fp16 runs, `fp32` for bf16 | 977.9 against 948.6 img/s with an fp32 wire, f16 KohakuVQ MLP decoder, batch 64, four GPUs |
| `CPU_THREADS` | 2 per rank | 338 to 677 img/s on four GPUs |
| `PRECISION` | `fp16` | same speed as bf16; see precision |
| `COMPILE_MODE` | `default` | `reduce-overhead` with the flat reducer: 222 against 224 img/s |
