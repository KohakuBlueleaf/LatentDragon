# Migrating legacy checkpoints

```bash
kogine run scripts/tools/convert_legacy.py --set SOURCE=old.ckpt --set TARGET=converted
```

Reads Lightning `.ckpt` and legacy `.safetensors`, handling key layout, EMA shadows, module
aliases, the `GruopNorm32` class path and hparams. Features that were not ported are refused
rather than dropped silently. `.ckpt` loading runs pickled code; only use your own files.

Going back, `scripts/export/export_run.py` writes `FORM=legacy_safetensors` for the legacy
tokenizer and `FORM=legacy_ckpt` for `LatentMaidVAE`. `FORM=diffusers` writes a folder a
diffusers `AutoModel` caller loads with `trust_remote_code`; the exported module imports
`latentdragon`, so it must be installed there.

Metric names are unchanged, except that per-channel `latent_stat_*` are no longer logged and
training PSNR, SSIM, `train/z_mean`, `train/z_std` and `vq/usage_ema` run on `METRICS_EVERY`.
Knob differences:

| Old | New |
| --- | --- |
| `GPUS` | `DEVICES` |
| `DISC` | `CRITIC` |
| `CODEBOOK_WEIGHT`, `KL_WEIGHT` | `AUX_WEIGHTS` (`kl` is 0 unless set) |
| `MODULE_EMA_DECAY`, `USE_EMA_MODULE` | `EMA_DECAY`, 0 turns it off |
| `EPOCH` | `EPOCHS` or `MAX_STEPS` |
| `LOG_INTERVAL`, `VAL_INTERVAL`, `CKPT_INTERVAL` | `SAMPLE_EVERY`, `VAL_EVERY`, `CKPT_EVERY` |
| `DDP_COMPRESS_HOOK` | `REDUCE_DTYPE` |
| `DDP_FIND_UNUSED_PARAMS` | gone, no DDP wrapper |
