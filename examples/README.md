# Examples

Runnable toys for layouts beyond one image in, one tensor latent, one image out. Every part
they add is registered from the example's own `parts.py`, the way an outside library adds its
own; the framework has no code for any of them. Each runs on the CPU in seconds:

```bash
python examples/paired_av/run.py
python examples/packed_latent/run.py
```

`checks/examples.py` runs both.

| Example | Data slots | Latent slots | Shows |
| --- | --- | --- | --- |
| [`paired_av`](paired_av) | `video`, `audio` | one `z` | a custom domain and source, two inputs fused into one latent, two outputs, per-slot losses |
| [`packed_latent`](packed_latent) | `x` (a clip) | `key`, `motion` | an encoder writing two slots, a `slots` latent def with `kl` and `vq` side by side, storing a packed latent and decoding it back |

## The pattern

1. A domain names the data slots, says how each is compared (`loss_dims`) and measured
   (`metrics`, `to_frames`), and turns a source item into a sample dict.
2. An encoder reads data slots and writes encoder slots.
3. A latent def serializes encoder slots into the latent, deserializes a latent, live or
   stored, into decoder slots, and packs the storable form.
4. A decoder reads decoder slots and writes the output latent.
5. A head reads the output latent and writes data slots; it names its objective.

Each part declares `inputs` and `outputs`, and the build checks the chain. `LOSS_CONFIGS` takes
terms per data slot, and `AUX_WEIGHTS` weighs the latent def's losses by name.
