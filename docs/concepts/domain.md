# Domain

A `Domain` is a frozen policy record. Tasks and eval call it rather than branching on image
versus video. `DOMAIN` holds `image` and `video`.

| Field | Meaning |
|---|---|
| `name` | Registry name |
| `rank` | Batch rank: 4 for `(B, C, H, W)`, 5 for `(B, T, C, H, W)` |
| `transforms(size, train)` | Sample pipeline: resize, crop, to tensor, normalize to [-1, 1] |
| `metrics` | Metric names eval should report |
| `to_grid(real, fakes, latents=None, limit=8)` | Sample grid as a uint8 CPU tensor |
| `to_frames(x)` | A frame batch for frame-wise metrics |

`image` transforms take a PIL image. `video` transforms take a `(T, C, H, W)` clip and draw
one crop for all frames.

`to_grid` stacks the rows real, latents, then each fake. `real` and `fakes` are in [-1, 1];
`latents` is an already rendered RGB tensor in [0, 1] at any size. The image grid is
`(H, W, 3)`; the video grid is `(T, H, W, 3)`. On video, latent frame 0 stays with the
keyframe and the later latent frames spread evenly over the later pixel frames.

Encoding a grid to a file stays with the logger.
