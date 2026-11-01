# Contributing to LatentDragon

Read this end to end before opening an issue or PR.

**English only.** Issues, PRs, commit messages and code comments are in English. Python source
and printed messages are ASCII-only.

Issues: https://github.com/KohakuBlueleaf/LatentDragon/issues. For a bug, include the run's
full log, its `config.json` and `git.json`.

## Before you open a PR

### Feature PRs need an approved issue first

A feature PR adds functionality, changes a public API or a knob, alters the architecture, or
changes user-visible behavior. Open an issue first and wait for a maintainer's explicit
approval. An issue and a PR opened at the same time does not count. PRs without that trail are
closed with a pointer here.

### Bug fixes and small changes can go straight to a PR

- bug fixes with a clear cause and no architectural change
- documentation, typo and dead-link fixes
- small refactors inside one module that keep every public API

### What the PR description holds

Use the PR template. At minimum: a summary, the measured problem, the exact validation commands
and the GPU they ran on, the linked issue, and any skipped check with its reason.

## Setup

```bash
git clone https://github.com/KohakuBlueleaf/LatentDragon && cd LatentDragon
uv venv -p 3.13
uv pip install -e ".[kernels,video,wandb,export,dev]"
```

Never use `sys.path` or `PYTHONPATH` hacks; scripts import the installed packages.

## Checks

There is no test suite and no CI. Behavior is verified by the scripts in [`checks/`](checks), run
by hand. [`checks/README.md`](checks/README.md) lists which script to run for which logic. State
the ones you ran in the PR. Numeric questions are reports from `experiments/`, read by a person.

Before pushing:

```bash
ruff check src checks scripts configs experiments
black --check src checks scripts configs experiments
python checks/repo_rules.py
```

## Code conventions

The full conventions are in [CLAUDE.md](CLAUDE.md). The non-negotiables:

- **Code carries only *what*; every *why* goes to `docs/`.**
- **Python 3.13**, `list`, `dict`, `X | None`, `match-case`. No `from __future__`.
- **Imports**: no imports inside functions, except guarded optional dependencies.
- **Files**: 600 lines, hard max 1000. One responsibility per module.
- **Select, don't dispatch**: configuration resolves once, at build time.
- **The model is pure**: it knows no loss. Tasks own losses; the engine owns stepping.
- **Eager ops are the reference**: a kernel matches them and is selected only where measured
  faster.
- **Public repository**: no machine names, hosts, users, local paths, credentials or private
  dataset details in tracked files or commit messages.

## Adding things

Most additions are a registry entry or a dotted path, not a core edit. See
[docs/guides/extending.md](docs/guides/extending.md). Any of these is still a feature contribution
and needs the approved issue first.

## License

LatentDragon is licensed under [Apache-2.0](LICENSE). By contributing, you agree that your
contributions are licensed under the same terms.
