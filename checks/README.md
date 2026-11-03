# Behavior checks

Plain scripts that verify execution behavior. Nothing runs them for you: when you change the logic
in the second column, run the script by hand and say so in the PR. All of them run on CPU with tiny
models and stop at the first failure.

| Script | Run it when you change | What it verifies |
| --- | --- | --- |
| `python checks/resume.py` | checkpoint, resume, RNG or loader state, driver step order, EMA | a crashed run ends identical to an uninterrupted one; a changed knob is refused on resume; a failed save keeps the last good checkpoint |
| `python checks/load.py` | arch files, state dict keys, converters, exports | every arch file builds; native, legacy and diffusers files rebuild the same model |
| `python checks/config.py` | knobs, config validation, recipes | every recipe is valid; bad knobs are reported together before device work |
| `python checks/reducer.py` | reducer, bundle, backward order | two ranks through each reducer give the gradients of one rank on the whole batch |
| `python checks/scaler.py` | loss scaler, checkpoint contents | the scaler backs off, grows and caps on the device, and a resume continues its count |
| `python checks/examples.py` | the slot contract, task, objective, latent defs | the paired video/audio and the keyframe/motion toys train, and a stored packed latent decodes like the live one |
| `python checks/state_terms.py` | latent terms, task states, conv head, sample callback | the state RMS term matches a direct computation and trains; an fp32 head skips the 16-bit cast; each sample step writes a train-batch and a fixed-batch grid |
| `python checks/unit_rows.py` | unit-row layers, the `mlp` or `res` block, norm `affine`, `init_writers` | rescaling raw rows leaves a unit-row block's output unchanged; gradients are tangent to the rows; only the writer has a gain; `writer_init="output"` moves to the gain; both block types train |
| `python checks/repo_rules.py` | package imports, or anything before a PR | file length, ASCII, in-function imports, import direction |

A change that touches none of this needs none of them; adding a module does not affect resume.

Numbers are not checked here. Decode strategies, precision, throughput and training results are
reports from [`experiments/`](../experiments) and from real runs, read by a person.
