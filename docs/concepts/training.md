# Training

| Owner | Owns |
| --- | --- |
| Task | modules, losses, seeds, metrics, what is saved |
| Driver | step order, micro-batch slicing, autocast |
| Bundle | optimizer, schedule, scaler, clip, reducer of one network |
| GanScheme | critic forwards and the composed backward |

One step is fixed: micro-steps, every bundle steps, EMA, `post_step`.

Why no Lightning: gradient control here is a seed-composed backward with per-network scale
state, deferred backward and batch-level adaptive weights. Each of those fought Lightning's
hooks. The loop is short enough to read, and it owns exactly the things the research
trainer patched around.

## Contract

```python
class Task:
    def groups(self) -> dict[str, list[dict]]: ...
    def micro_step(self, ctx: MicroCtx) -> dict: ...
    def saved_modules(self) -> dict[str, Module]: ...
```

`ctx.backward[name]` routes a scalar through that bundle's scaler. `ctx.seed_scale` is the
model bundle's scale, for seeds taken with `seed_of`.

## Distributed

Gradients are averaged once, in `Bundle.step`, after accumulation. There is no DDP wrapper,
so there is no `find_unused_parameters`, no buffer broadcast and no `no_sync` rule: a scheme
may defer its backward freely. Mutated state is shared explicitly in `Task.post_step`;
today that is a stat-tracking critic's BatchNorm buffers, broadcast from rank 0. EMA shadows
live outside every module and are never broadcast.

## Selection

`NoScaler`, `NoEMA`, `NoReducer` and `NoUsage` replace per-step checks. Gradient
checkpointing and compile are applied at build time and keep state keys unchanged.
