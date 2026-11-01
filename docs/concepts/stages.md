# Stages and heads

The decoder splits into a trunk and a head:

```text
input -> encode -> latent -> trunk -> output latent -> head -> output
```

Stage one trains everything with the regression head. A later stage holds what should stay
fixed and retrains the rest. The tokens do not change, so generators trained on them stay
valid. Stages are separate runs linked by `INIT_FROM`, never one run with a phase switch.

```python
INIT_FROM = "runs/stage1/<id>/checkpoints/step_00400000"
FREEZE = ["encoder", "bottleneck", "trunk"]
```

Freeze depth is a knob: `"trunk.stages.0"` holds one stage. Frozen parts stay in eval mode
and never enter a bundle. A frozen encoder runs under `no_grad`; a bottleneck and trunk do too
once everything before them is held, so a live encoder still gets its reconstruction gradient.

`INIT_FROM` reads live weights with EMA laid on top, reports loaded, skipped and missing
counts, and refuses to leave a frozen part unset. A staged run resumes with `RESUME`;
`INIT_FROM` is skipped once a checkpoint exists under `RESUME`.

Why the split: at VQ compression much of the output is generated, not reconstructed, and
adversarial training has not been shown to help the latent. Separating them lets one latent
serve many heads. Only `regression` exists today. A new head registers in `HEAD`, pairs with
an objective in `HEAD_OBJECTIVES`, and reads trunk features. `needs_pixels` is the slot for
a refiner: the task passes `pixel_source(features)` to the head. No knob configures a source
yet, so config checks refuse such a head.
