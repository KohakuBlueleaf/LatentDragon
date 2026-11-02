# LatentDragon

Image and video latent autoencoder framework: VQ and continuous bottlenecks, GAN schemes,
staged heads and latent slots, on a selection-based trainer. Sibling to
KohakUwULLM and KohakuMoss; same house conventions.

## Packages

| package | role | imports |
| --- | --- | --- |
| `latentdragon_common` | registry, build, imports, log, knobs, null callables | no torch, no sibling |
| `latentdragon_kernels` | ops, registry, eager, gemm and Triton backends, tuning, error stats | common |
| `latentdragon_data` | sources, epoch-resumable loader | common |
| `latentdragon` | model, bottleneck, vq, latent, gan, objectives, tasks, training, eval, inference, io | all |

`model`, `vq` and `bottleneck` never import `training`. `checks/repo_rules.py` verifies
direction, file length, ASCII and in-function imports.

## Code conventions

- Max 600 lines per file, hard max 1000. One responsibility per module.
- Code carries only *what*; every *why* goes to `docs/`. Docstrings 1-3 lines, comments one
  line. No rationale, measurement or history in source.
- No imports inside functions except try-guarded optional deps.
- Python 3.13. `list`, `dict`, `X | None`, `match-case`. No `from __future__`. ASCII-only
  source and messages.
- `black` and `ruff check` over `src checks scripts configs experiments`.
- Sentence-case imperative commits; body gives the problem, then the change.
- Public repository: no machine names, local paths, credentials or private dataset details
  in tracked files. Local notes in `CLAUDE.local.md`, plans in `plans/` (gitignored).

## Architectural rules

**Select, don't dispatch.** Configuration resolves once at build time via
`build(choice, REGISTRY)`. High-multiplicity code has no per-step mode branch; a branch the
loop takes once per step is fine when it reads better.

**The model is pure.** `Autoencoder` is encoder, bottleneck, trunk and head, each callable.
It knows no loss. Objectives pair with heads.

**Tasks own losses; the engine owns stepping.** A task never steps an optimizer, a driver
never computes a loss. Each network has its own `Bundle`.

**Eager ops are the reference.** Every backend equals `backends/eager` within the precision
bar; a kernel is selected only where measured faster.

## Numerics

- bf16 is the default. fp16 needs seeds from scaled losses, per-bundle scalers and
  unscale-before-clip.
- Reduce in fp32. fp64 exists only as a reference in experiments.

## Checks and experiments

This repo has no test suite and no CI workflow. Nothing here runs on every commit.

- `checks/` holds behavior scripts, run by hand when the logic they cover changes.
  `checks/README.md` says which script goes with which logic.
- `experiments/` holds numerical questions. They write JSONL for a person to read, with no
  pass/fail threshold: precision depends on relative and absolute error, stats and distribution.
- Results are judged end to end, on real runs.
- Verify a change once, with the check it touches or a one-off run, then stop. A new feature
  needs no new check by default.

## Workflow

1. Implement the whole slice.
2. Audit the diff for clear, integrity and behaviour bugs, and for cost.
3. `black`, `ruff`, then the `checks/` scripts the change touches.

## Setup

```bash
uv pip install -e ".[kernels,video,wandb,export,dev]"
kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_kvq_f16c256.py
```

Never use `sys.path` hacks; import the installed packages.
