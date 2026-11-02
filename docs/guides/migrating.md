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

Metric names follow the research trainer for the headline values (`train/eloss`, `train/psnr`,
`train/ssim`, per-channel `latent_stat_*`). Every other scalar is logged raw, with an `_ema` twin
from the trainer's own smoothing; the code-usage window is `vq/usage_window`. Knob differences:

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
| `ENC_LR_SCALE`, `VQ_LR_SCALE`, `DISC_LR_SCALE` | `LR_SCALES` by group: `encoder`, `latent`, `decoder`, `head`, `critic` |

## Earlier LatentDragon runs

Runs and exports written before the slot contract named the parts `bottleneck` and `trunk` and
kept a flat `ARCH` (`latent_dim`, `encoder_blocks`, `trunk`, ...). `latentdragon.io.arch_json`
`upgrade_arch` turns a flat arch into the part form, and checkpoints, `INIT_FROM` weights and
exports load their old keys under `latent.` and `decoder.`. `checks/load.py` loads such an
export.
