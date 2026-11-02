# Domain

A `Domain` is the data def: it names the slots of a sample and owns everything that reads them.
Tasks and eval call it rather than branching on image versus video. `DOMAIN` holds `image` and
`video`, each with one slot, `x` by default (`{"name": "image", "slot": "rgb"}` renames it).

| Field | Meaning |
|---|---|
| `name` | Registry name |
| `slots` | The data slot names |
| `transforms(size, train)` | A callable from a source item to a sample dict |
| `loss_dims` | Per compared slot, the sensecraft layout: 2 for frames, 3 for clips |
| `metrics` | Per slot, the names among `psnr`, `ssim` and `rfid` |
| `to_frames` | Per metric slot, a flattener to `(N, C, H, W)` in [-1, 1] |
| `to_grid(real, fakes, latents)` | The sample picture as a uint8 CPU tensor, or None for no picture |
| `save_grid(grid, path)` | Writes that picture |

`image` transforms take a PIL image; `video` transforms take a `(T, C, H, W)` clip and draw one
crop for all frames. Sources call the transform, so a source's item becomes a sample dict.

`to_grid` stacks the rows real, latent preview, then each fake. `real` and `fakes` are sample
dicts in [-1, 1]; `latents` is a channels-first RGB preview in [0, 1] at any size. The image grid
is `(H, W, 3)`; the video grid is `(T, H, W, 3)`, with latent frame 0 at the keyframe and later
latent frames spread over the later clip frames.

A domain with several slots, like the paired video and audio toy in `examples/paired_av/`,
lists per slot what applies: losses by `LOSS_CONFIGS[slot]`, metrics by `metrics[slot]`, and
validation names become `val/<slot>/<metric>`.
