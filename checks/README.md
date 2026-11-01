# Behavior checks

Plain scripts that verify execution behavior. Nothing runs them for you: when you change the logic
in the second column, run the script by hand and say so in the PR. All of them run on CPU with tiny
models and stop at the first failure.

| Script | Run it when you change | What it verifies |
| --- | --- | --- |
| `python checks/resume.py` | checkpoint, resume, RNG or loader state, driver step order, EMA | a crashed run ends identical to an uninterrupted one; a changed knob is refused on resume; a failed save keeps the last good checkpoint |
| `python checks/load.py` | presets, state dict keys, converters, exports | every preset builds; native, legacy and diffusers files rebuild the same model |
| `python checks/config.py` | knobs, config validation, recipes | every recipe is valid; bad knobs are reported together before device work |
| `python checks/repo_rules.py` | package imports, or anything before a PR | file length, ASCII, in-function imports, import direction |

A change that touches none of this needs none of them; adding a module does not affect resume.

Numbers are not checked here. Precision, throughput and training
results are reports from [`experiments/`](../experiments) and from real runs, read by a person.
