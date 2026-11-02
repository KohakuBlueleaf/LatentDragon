# Inference

`latentdragon.inference` decodes latents that do not fit at once. A strategy is selected by name,
like every other choice in the framework:

```python
from latentdragon.inference import decode, stream_encode

out = decode(model, latent)                                     # the whole latent at once
out = decode(model, latent, {"name": "tiled", "tile": [512, 512]})
out = decode(model, packed, "streaming")
```

`latent` is a live or packed latent; the latent def reads it into the decoder's slot, and the
result comes back under the head's output slot. Strategies tile that one decoder input, so they
need a single-slot decoder and head with slot-local `tensor` calls. Sizes in a spec are output
pixels and frames; overlaps are latent cells.

| Strategy | What it does | Needs |
| --- | --- | --- |
| `full` | decodes the whole latent | |
| `tiled` | decodes spatial tiles, and spans of latent frames when `tile_t` is set; `inner` decodes each tile | a bounded reach, or `overlap` |
| `staged_tiled` | runs the first `stages` decoder stages whole, then tiles the rest | `forward_early`, `forward_late`, `scales` |
| `streaming` | decodes chunks of latent frames, holding only the causal cache | `forward_stream` |
| `streaming_tiled` | `streaming` with spatial tiles, one cache per tile | `forward_stream` |

`tiled` with `inner={"name": "staged_tiled", ...}` gives two-stage tiling: outer tiles of the
latent, each decoded by staged tiling. `streaming` keeps memory constant with clip length;
`offload_cache` holds the caches on the CPU between chunks. `StreamingDecode.stream` yields frames
as each chunk finishes. `stream_encode(model, clip)` does the same for the encoder and yields its
output, before the latent def.

## Reach

A tile is exact when it is fed every cell its kept region depends on. `model.reach()` returns that
distance as a `Reach`: `spatial` cells on each side and `temporal` past frames, on the latent grid.
It is summed from the model: each `Block` declares `spatial_reach` and `temporal_reach` at its own
resolution, and the decoder scales them by where the block sits. A decoder that declares no
`reach` has no exact overlap. `model.reach(start)` gives the
reach of the stages from `start` on that stage's grid, which is what staged tiling uses.

Spatial tiles are fed `reach.spatial` cells past each kept edge. Temporal tiles are causal: each is
fed `reach.temporal` past frames and keeps the frames after them.

`spatial` is None when the path holds attention, group norm or a block that declares no reach,
because those read every position. `tiled` then refuses unless `overlap` is set, and with
`overlap` the result is an approximation that can show seams.

A strategy's `requires` lists the decoder methods it calls, and `check` reports what is missing
before any decoding starts. A new decoder gains a strategy by providing those methods, and a new
strategy registers in `DECODE`.

`experiments/decode/strategies.py` prints each strategy's difference from `full`, with controls
that use too little overlap.
