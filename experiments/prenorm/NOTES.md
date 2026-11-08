# Free scales and unit rows in pre-norm residual nets

These are notes from one design discussion. They cover where magnitude drift comes from in
`x = x + f(norm(x))` networks, which fixes were tried, what a CPU toy measured, and what went into
LatentDragon.

The toy is [`toy_drift.py`](toy_drift.py). Its results are written to `runs/prenorm/toy_drift.json`.
The interactive page is https://claude.ai/artifact/APsycyKYncqj6rsWsinF5q.

Everything here is one seed, one small model, and 3000 steps. Read the numbers as directions, not
as settled results.

---

## 1. Where it started

The production run (LatentMaid encoder → KohakuVQ → LatentMaid decoder → ConvHead) showed two
symptoms:

- **With no encoder-side loss, the encoder output kept growing.** The cause is the VQ's RMS norm.
  The quantizer only sees `rms_norm(h)`, so the loss is invariant to the scale of `h`. Its gradient
  is then perpendicular to `h`, and every finite step makes `‖h‖` larger. Setting the latent's
  `"normalize": None` turns that norm off. The commitment and codebook loss then pin the scale.
- **The decoder's residual stream grew while the head weights barely moved.** The head reads the
  stream directly (`y = W r`), so the loss only constrains the product `‖W‖·‖r‖`. Under Adam, the
  relative drift of a weight is about `η√t / σ_init`. The tiny-init writers (σ ≈ 1e-3) drift about
  100× faster in relative terms than a head at σ ≈ 0.1. So the stream ends up taking the scale.

### Head init: `preserve_init`

A magnitude-preserving head init pins the head/stream split where the init puts it. The default
init leaves that split to drift, because its std is `1/fan_in` (or 0 with `zero_init`).

- **Shrink pressure is short-lived.** For an MSE head, the radial gradient is
  `⟨∇W, W⟩ = E‖Wr‖² − E⟨x, Wr⟩`. It is positive only while the output is uncorrelated with the
  target, and it goes to 0 once the output fits.
- **The head shrinks only a little.** At std 0.125 and η = 1e-4, coherent shrinking would take
  about 1000 steps. The stream fixes the output within tens to hundreds of steps, so the head loses
  a few percent at most.
- **After that, the init decides the split.** At 10k steps the drift is about 8% for this init. It
  is unbounded for `zero_init`.

`ConvHead(preserve_init=True)` keeps the hidden-layer std `fan_in^-0.5` and starts the bias at 0.
On CPU, a unit-RMS input gives an output RMS of 0.97, against 0.16 for the default init. The
next run's residual scale was "way more reasonable". (Commit `730865c`.)

`standardize=True` is the stricter version: unit rows with no gain, so the head cannot shrink at
all.

---

## 2. The mechanism: free scales

**Definition.** A free scale is a parameter group `g` with `L(c·g) = L(g)` for all `c > 0`.
Differentiating at `c = 1` gives `⟨g, ∇g L⟩ = 0`: the gradient is always perpendicular to `g`. Any
finite step then grows the norm:

```
‖g'‖² = ‖g‖² + η²‖∇‖²        (EDM2 Eq. 62–65; Kunin et al., "Neural Mechanics", ICLR 2021)
```

Nothing pulls it back. Weight decay is the classic counterweight, and LD runs with
`WEIGHT_DECAY = 0`.

**How fast it moves under Adam.** Adam moves each entry by about η per step, so a group drifts by
about `η/σ` relative to its size per step. Small-init groups drift the most. For a coherent rank-1
gradient the update is `η·sign(g)sign(u)ᵀ`, whose spectral norm is `η√(mn)`. That is the fast
blow-up right after warmup.

**Two kinds of growth.**

| Kind | Cause | Removable by design? |
|---|---|---|
| (a) symmetry drift | the loss cannot see the scale, so it random-walks up | yes |
| (b) functional growth | the network wants a large value (attention sinks, massive activations) | no; it can only be routed through visible owners |

### Every free scale in a standard pre-norm block

A free pair exists wherever two parameters meet through a homogeneous op.

| Pair | Why it is free | What grows |
|---|---|---|
| norm gain γ ↔ next linear's rows | `W(γ⊙u) = (W·diag γ)u` | the gain or the weights |
| SwiGLU `W_up` ↔ `W_down` | the up branch is linear | the MLP hidden state |
| ReLU MLP `W1` ↔ `W2` (GELU/SiLU approximately, once large) | homogeneity | the hidden state |
| `W_q` ↔ `W_k` | logits are `uᵀW_qᵀW_k u` | the logits, then attention entropy collapses |
| `W_v` ↔ `W_o` | attention is linear in v | the value states |
| (`r0`, all writers) jointly | every stream reader normalizes | the residual stream |

**The stream's joint scale.** In a pre-norm net every reader of the stream (each pre-norm and the
final norm) is scale-invariant. So scaling `r0` and every writer by the same `c` is free. This is
why removing DiT's final norm helps: the head then reads `r` raw, and the loss pushes back.

Removing the final norm doesn't fix everything:

- the loss only sees `W r`, so the null space of `W` stays free (about 1120 of 1152 directions for
  DiT);
- the head and the stream can still trade scale;
- kind (b) growth is untouched.

---

## 3. Unit rows

**Parametrization.** Each output row is normalized at use, `ŵ_j = w_j/‖w_j‖`. All scale lives in
explicit per-channel gains.

**Bounded by construction.** For a reader of a normalized input (`‖u‖ = √d`), Cauchy–Schwarz gives

```
|h_j| = |⟨ŵ_j, u⟩| ≤ √d
```

Every pre-activation is bounded. Following the bound through the block:

- the activation and the SwiGLU product are bounded;
- a unit-row writer on a bounded hidden state gives a bounded write;
- attention gives a convex combination of bounded values.

So the stream grows at most linearly with depth and no longer with training time.

**No capacity is lost.** `w = g·v̂` represents the same functions as a plain matrix (Salimans &
Kingma). Only the place where the scale lives changes.

**Raw weights still grow.** Under normalize-at-use the raw weights are scale-invariant, so they
still grow (EDM2 Eq. 62–65). The function doesn't see it. Its only effect is an effective learning
rate that decays like `1/‖w‖`.

### "No free move" is not "no move"

- **Scale can still change, through gains.** Every change has to go through a gain, the loss sees
  gains, and they move only when the loss asks.
- **No bounded input or output is needed.**
  - Readers read a norm's output.
  - Writers read hidden states that the readers already bound.
  - Only the input projection reads from outside the network, and it needs a *stable* input scale
    (ordinary standardization), not a bounded one.
  - The output is bounded only by `gain × input norm`, and the gain is unbounded. An LM's logit
    temperature is just the head gain.
- **What changes is where the growth shows.** If the loss wants unbounded outputs, the growth shows
  up in one loggable gain instead of spreading across every matrix.

### Free directions that matter

The goal is not EDM2's "no free direction at all". It is **no free direction that moves
activations**. Free directions come in two kinds:

- **Activation-moving: dangerous.** Some intermediate tensor changes scale while the loss stays the
  same: up↔down, v↔out, raw growth of any plain matrix, the stream's joint scale.
- **Parameter-only: harmless.** Every activation stays the same: the raw norm behind a unit-row
  matrix, and `γ_q ↔ γ_k` in QK-norm. At worst they shift an effective learning rate.

That defines the "important stuff" to make unit-row: **every matrix whose raw scale reaches an
activation, which is every matrix inside the residual blocks.** The network's ends can stay
conventional.

### Unit rows are one instance of a general rule

The rule: every matrix's raw scale must be either invisible to the function or held by something.

| Where | Option | Bounds | Tested |
|---|---|---|---|
| weights | unit rows (EDM2 weight norm) | every pre-activation, with no hyperparameter | **yes, holds** |
| | weight standardization | the same | writers only; behaved like unit rows |
| | Frobenius norm per matrix | total output scale only | no |
| | spectral norm (σReparam) | `‖Wu‖ ≤ γ‖u‖` | no |
| activations | post-norm per write (Peri-LN, sandwich) | the stream, but not the inside of f | no |
| | a norm per internal site | that site only (QK-norm) | QK-norm only |
| optimizer | weight decay | raw norms reach an equilibrium | no |
| | retraction (EDM2 forced WN) | fixed effective learning rate | yes (§6) |
| | Muon | the growth *rate* only | **yes, slows the growth, doesn't stop it** |

A post-norm on each write is the mirror image of unit-row writers. It bounds the stream but leaves
`W_up`/`W_v` free, so f's hidden states can still explode, which is an fp16 overflow concern.

---

## 4. Rejected: a capped reader (and DyT)

**The proposal.** Use `norm_C(x) = norm(C·tanh(x/C))` as a soft ceiling on the readers, so that
drift costs loss without a new loss term.

**Why it was rejected.**

- **It adds a hyperparameter.** `C` is a clamp in soft form, and the right value depends on depth,
  width and outliers ("clamp value introduce tons of new hyperparam").
- **It doesn't restore anything.** An overshoot lands deep in saturation, where `tanh' → 0` and no
  gradient pulls the stream back.
  - Cap alone at lr 3e-2: stream RMS about 1.2e6, and the loss was *worse* than baseline (3.13 vs
    2.47).
  - Cap + unit rows everywhere at 3e-2: stream RMS 26 against C = 8. The readers were saturated,
    so something else was doing the bounding.
- **Without the cap, unit rows hold just as well.** Its effect on top of them was a few hundredths:
  slightly better at low learning rates and worse at 3e-2.

**DyT has the same problem.** `γ·tanh(αx)` with a learnable `α` brings the free scale back (stream
×c, α/c). Its own paper says `α₀` needs tuning for LLMs.

---

## 5. Toy results: what bounds what

**Setup.** A byte-level LM on the Python stdlib source, run on CPU:

- 6 pre-norm blocks, width 128, 4 heads, SwiGLU hidden 384;
- context 64, batch 32, 3000 steps, 100 warmup;
- Adam (0.9, 0.95), no weight decay, one seed.

The table shows held-out loss, with the final stream RMS in parentheses. "Explodes" means a stream
RMS in the thousands or more.

| Variant | lr 3e-3 | lr 1e-2 | lr 3e-2 |
|---|---|---|---|
| baseline (plain) | **1.340** (13) | 2.171 (6.4e3) explodes | 2.466 (2.2e6) explodes |
| cap only | 1.361 (14) | 2.950 (1.4e4) | 3.129 (1.2e6) |
| unit-row writers | 1.369 (4.4) | 1.397 (14) | 2.487 (3.5e5) explodes |
| standardized writers | 1.345 (4.4) | 1.396 (14) | 2.423 (3.9e5) explodes |
| unit-row writers + fixed-scale input | 1.341 (4.4) | 1.385 (13) | 2.669 (1.9e5) explodes |
| cap + unit-row writers | 1.377 (4.6) | 2.196 (90) | 3.015 (9.9e4) |
| cap + unit rows everywhere | 1.350 (3.3) | 1.366 (4.6) | 1.423 (26) |
| cap + unit rows everywhere + QK-norm | 1.344 (3.3) | 1.327 (4.7) | 1.369 (12) |
| unit rows everywhere | 1.363 (3.3) | 1.354 (4.3) | 1.356 (7.8) |
| **unit rows everywhere + QK-norm** | 1.355 (3.3) | 1.352 (4.8) | **1.339** (9.1) |

In the "everywhere" variants the embedding has unit-RMS rows (a fixed-scale input), and the head is
unit rows plus a gain.

**Peak values at lr 3e-2:**

| | stream max | MLP up max | attention logit max |
|---|---|---|---|
| baseline | 3.5e7 | 2300 | 3.2e5 |
| unit rows everywhere + QK-norm | 231 | 18 | 20 |

**Findings.**

- **The baseline explodes above lr 3e-3.** The blow-up starts right after warmup: the stream RMS is
  about 2900 by step 300.
- **Writers alone are not enough**, whether unit-row, standardized, or with a fixed input. The
  readers' raw growth still flows into f's hidden states, so the writer is a bounded map on an
  unbounded input.
  - With only the writer's own scale removed, the joint free scale moves onto `W_up` and the gains.
- **Unit rows on every block matrix plus QK-norm bounds the stream and f's internals together.**
  - It reaches 1.339 at lr 3e-2, matching the baseline's best at 10× its learning rate.
  - It is neutral at the baseline's best learning rate (1.355 vs 1.340).
- **Adam grows the raw norms about equally with or without unit rows** (gate/up summed norm:
  28 → 619 baseline, 28 → 481 unit rows).
  - In a plain net that growth *is* activation growth.
  - Under unit rows it only decays the effective learning rate.
- **The bounding comes from the anchor, with no new constant.** With unit-row writers the stream's
  joint scale becomes (`r0`, writer gains). With `r0` at a fixed scale, scaling the gains changes
  how `r0` and the writes mix, and the loss sees that.

### Adam vs Muon

| Held-out loss | lr 3e-3 | lr 1e-2 | lr 3e-2 |
|---|---|---|---|
| Adam, baseline | 1.340 | 2.171 explodes | 2.466 explodes |
| Muon, baseline | 1.335 | **1.300** | 2.045 explodes |
| Adam, unit rows + QK | 1.355 | 1.352 | 1.339 |
| Muon, unit rows + QK | 1.319 | 1.308 | 1.335 |

Muon was used on the block matrices, with Adam on the embedding, gains and head. Its update was
scaled to match Adam's RMS.

- **Muon caps the per-step spectral growth but has no restoring force.**
  - At 1e-2 it is the best run, but its stream is still climbing (RMS 20 → 79).
  - At 3e-2 it explodes.
  - This matches reports at scale: Moonlight needed weight decay, and Kimi K2 added MuonClip, a
    tuned threshold.
- **With unit rows, Muon is stable at every learning rate** and slightly better than Adam.
- **Under unit rows, Muon grows the raw weights faster** (gate/up 28 → 927 vs 481), because its
  full-rank updates add more to `‖W‖²`. Its effective learning rate therefore decays faster.
- **In short:** unit rows remove the path the growth takes, and Muon slows the rate.

---

## 6. Three ways to keep unit rows

| | normalize at use | projection only | both (EDM2) |
|---|---|---|---|
| forward | `F.linear(x, w/‖w‖)` | `F.linear(x, w)` | `w/‖w‖` |
| tangent gradient | automatic: `(1/‖w‖)(I − ŵŵᵀ)∇ŵL` | project `grad` before the step | automatic |
| after the step | nothing | retract rows to norm r, keep `exp_avg` tangent | retract |
| effective lr | decays like `1/‖w‖` (implicit schedule) | fixed, set by the schedule | fixed |
| weight decay | manages the effective lr | meaningless (retraction undoes it) | meaningless |
| EMA, loading, manual edits | always correct | must renormalize | always correct |

Notes on projection:

- **The tangent pre-projection is required.** Without it, the radial component inflates Adam's
  second moment, and the retraction throws its effect away (EDM2 Fig. 23).
- **The retraction norm sets the learning-rate scale.** At norm 1, entries are about `1/√n`, so an
  η-per-entry step is a relative change of about `η√n` per row (11–20× here). EDM2 keeps rows at
  `√fan_in` and folds a fixed `1/√n` into the forward.
- **EMA was set aside.** An average of unit vectors has norm below 1, but EMA doesn't matter for
  now.

**Results** (unit rows everywhere + QK-norm, rows retracted to norm 1; held-out loss):

| lr | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|
| baseline | 1.492 | 1.344 | **1.340** | 2.171 | 2.466 |
| normalize at use | 1.633 | 1.406 | 1.355 | 1.352 | **1.339** |
| projection only | 1.633 | **1.395** | 1.429 | 1.646 | 2.033 |
| both | 1.633 | **1.396** | 1.420 | 1.644 | 2.033 |

- **Projection and "both" are nearly identical,** which means the tangent pre-projection is doing
  its job.
- **Both stay bounded at every learning rate** (stream RMS ≤ 3.2), and their raw norms stay
  constant (gate/up 27.7).
- **They trail normalize-at-use at higher learning rates** because nothing decays their effective
  learning rate. Normalize-at-use gets a decay for free from raw-norm growth.
  - A fair comparison needs a decaying schedule (EDM2 uses inverse-sqrt). That has not been run.
- **Normalize-at-use is the cheapest choice:** no optimizer change, and EMA and loading stay
  correct. That's why LD uses it.

---

## 7. Affines, the ends, and owner options

**Rule: an affine only where it is the sole owner of a scale.**

| Norm / gain | Affine? | Why |
|---|---|---|
| pre-norm before a unit-row reader | **no** | the reader's direction already covers per-channel weighting; the overall size pairs with downstream gains (v→out, up→down) |
| input norm (after the input projection) | **no** | a gain there trades with all writer gains (the stream's joint scale) |
| final norm before a head with a gain | **no** | it pairs with the head gain |
| QK-norm | **one side only** (on q, or one temperature per head) | with no gain, logits are capped near `√d_head` (5.7 at 32); with gains on both sides, `γ_qγ_k` is a parameter-only free pair (the toy had both sides, harmlessly) |
| writer | **per-channel gain** | the only thing that sets how much a block writes |
| gate branch before SiLU | optional gain, **untested** | SiLU isn't homogeneous, so nothing else can move its operating point |
| norm bias | no | Adam can push it about η per step without limit; EDM2 dropped every bias |

**Gains and biases are good owners.** A per-channel gain's gradient sums over every position, so it
is low-noise. Under a consistent preference it moves about `ηt`, while noisy matrix entries move
about `η√t`. It tracks the loss quickly and visibly. The flip side: functional growth (sinks,
sharper logits) arrives fast through gains, but it stays visible.

**Massive activations in MOSS-image.** The extreme residual there is written by the first block.
That matches "massive activations" (Sun et al. 2024), which is kind (b) growth. No free-scale rule
removes it. Unit rows only force it through visible parameters: the first writer's gain and its
bounded hidden state.

### Input and output owner options

There are two axes:

- who anchors the stream: a plain input projection, an input norm without affine, or an input norm
  with affine;
- who owns the output scale: a head gain, the final norm's affine, or the stream itself.

That makes nine combinations. Four are sensible:

| | Input | Output | Owner of output scale | Fits |
|---|---|---|---|---|
| A | norm, no affine | norm, no affine → unit rows + gain | head gain (per output) | general; the toy used this |
| B | no input norm | norm, no affine → unit rows + gain | head gain; one global free stream scale remains | probably fine (one slow direction), worth measuring |
| C | norm **with** affine | no final norm → unit rows, no gain | the stream itself, pinned by the loss | regression / AE; bad for CE (temperature lives in the stream magnitude) |
| D | norm, no affine | norm **with** affine → unit rows | final-norm γ (per stream channel) | CE (temperature in γ, compatible with weight tying); RGB scale control is slightly entangled |

**A rejected combination:** an input norm with affine, a final norm without affine, and a head
gain. The input gain and the writer gains trade freely.

**Input amplitude that carries information.** If the input's amplitude means something (audio
loudness, an AE decoder reading a latent), don't normalize it. Let it reach `r0` through a unit-row
projection, where it stays visible as the `r0`-to-writes ratio.

A/B/C/D have not been compared in the toy.

---

## 8. Finding free scales statically

**Degree propagation (dimensional analysis on the graph).**

1. Give each parameter group `k` a log-scale `s_k`. Every tensor then gets a degree vector `d`:
   scaling the parameters by `e^s` scales the tensor by `e^{d·s}`.
2. Propagate degrees through the graph, collecting a constraint wherever a scale must stay fixed:

   | Op | Output degree | Constraint |
   |---|---|---|
   | plain linear/conv `k` | `d_in + e_k` | — |
   | unit-row linear | `d_in` | — |
   | gain `k` | `d_in + e_k` | — |
   | `a ⊙ b` | `d_a + d_b` | — |
   | `a + b` | `d_a` | `d_a·s = d_b·s` |
   | norm, no affine | 0 | — |
   | ReLU | `d_in` | — |
   | SiLU / GELU / tanh | 0 | `d_in·s = 0` |
   | attention | `d_v` | `(d_q + d_k)·s = 0` |
   | loss input | — | `d_out·s = 0` |

3. **The free scales are the null space of the stacked constraints.** Each basis vector names which
   parameters trade.
4. **A free direction is dangerous when some intermediate tensor's degree along it is nonzero**,
   i.e. it moves activations.

This can be implemented with `torch.fx`, with unit-row layers, gains and norms as leaf rules.

**What the static check misses:**

- approximate symmetries (SiLU ≈ ReLU at large inputs);
- the *strength* of weak pins, such as the `r0` ratio fading as writes grow;
- non-scale symmetries.

**The numeric complement is the scale-Jacobian spectrum.** Compute
`J_mk = ⟨∇θ_k(r_mᵀ f(x)), θ_k⟩` with M random output projections, then take the SVD of `J`.

- Zero singular values are exact free directions.
- Small singular values are near-free directions, and their size shows how strongly each one is
  pinned.

**Predictions:**

- the baseline has many free directions;
- unit rows + QK-norm leaves only the per-dimension `γ_q ↔ γ_k` pairs (parameter-only);
- B has one global direction.

Not built yet (planned as `experiments/prenorm/free_scales.py`).

---

## 9. Pseudocode: unit rows everywhere

**The building block.** Everything uses one primitive: a linear or conv with rows normalized at
use, plus an optional per-channel gain. Each layer takes one of three roles:

- **Reader** (right after a norm): unit rows, no gain, and the norm in front has no affine.
- **Writer** (adds into the stream): unit rows plus a gain.
- **Stream-to-stream map** (shortcut, resample): unit rows, no gain.

```python
class SwiGLUBlock(nn.Module):                     # x + down(silu(gate) * up)
    def __init__(self, dim, hidden):
        self.norm = RMSNorm(dim, affine=False)
        self.gate_up = UnitRowLinear(dim, 2 * hidden)            # reader
        self.down = UnitRowLinear(hidden, dim, gain=True)        # writer
    def forward(self, x):
        gate, up = self.gate_up(self.norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)

class Attention(nn.Module):
    def __init__(self, dim, heads):
        self.norm = RMSNorm(dim, affine=False)
        self.qkv = UnitRowLinear(dim, 3 * dim)                   # reader
        self.q_norm = RMSNorm(head_dim)                          # gain = temperature (one side)
        self.k_norm = RMSNorm(head_dim, affine=False)
        self.out = UnitRowLinear(dim, dim, gain=True)            # writer

class ResnetBlock(nn.Module):                     # SD-VAE style
    def __init__(self, c_in, c_out):
        self.norm1 = GroupNorm(c_in, affine=False)
        self.conv1 = UnitRowConv2d(c_in, c_out, 3)               # reader
        self.norm2 = GroupNorm(c_out, affine=False)
        self.conv2 = UnitRowConv2d(c_out, c_out, 3, gain=True)   # writer
        self.skip = UnitRowConv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
```

**Initializing the writer gains.** Use the depth rule `1/√(2L)`, or keep the existing writer init:
`writer_init="output"` (std `1/fan_in`) is exactly a gain of `fan_in^-0.5` on unit rows.

**Don't add an epsilon to the norm.** The first draft divided by `‖w‖ + 1e-4`. With rows near
norm 1, that made the output depend on the raw scale (error 1.5e-3). Use a floor that only guards
an all-zero row, `clamp_min(1e-12)`. Compute the norm in at least fp32, using `promote_types`,
never `.float()`, because `.float()` downcasts fp64.

### A fused kernel (not built)

Normalizing the weight before the GEMM costs an extra launch and about three weight-sized memory
passes. Row normalization is a per-output-channel scale, so it can be fused:

```
y_j = s_j · (W x)_j,     s_j = g_j / ‖w_j‖
```

| Piece | Where |
|---|---|
| row norms `‖w_j‖` | once per step, folded into the optimizer's weight-write pass |
| `s_j` in the forward | GEMM/conv epilogue (like int8/fp8 dequant), or fused into the next elementwise op |
| dgrad `Wᵀ(s ⊙ dy)` | prologue scale on dy, or folded into the elementwise backward that made dy |
| wgrad | the unchanged GEMM gives the raw `R = dyᵀx` |
| chain rule | optimizer pre-step: `dot = ⟨ŵ_j, R_j⟩`, `∂g = dot`, `∂w = s_j(R_j − ŵ_j·dot)` |

The math is the same as normalize-at-use, laid out the way the projection approach would lay out
its kernels. Retraction becomes a flag in the post-step. LD uses plain torch plus `torch.compile`
for now.

---

## 10. What went into LatentDragon

**Scope: blocks only.** The flag covers the LatentMaid `mlp` block (and `mlp_plain`) and the SD-VAE
`res` block. Stage projections, attention slots, the encoder's `out_proj` and the head are
untouched. Biases are kept for now.

```python
"block": {"name": "mlp", "unit_rows": True}      # or {"name": "res", "unit_rows": True}
```

| Block | Layer | With the flag on |
|---|---|---|
| `mlp` | `norm` | no affine |
| | depthwise mixer | unit rows (per-channel 3×3), no gain |
| | `swiglu.inp` | unit rows, no gain |
| | `swiglu.out` (writer) | unit rows + per-channel gain |
| `res` | `norm1`, `norm2` | no affine |
| | `conv1`, 1×1 `skip` | unit rows, no gain |
| | `conv2` (writer) | unit rows + per-channel gain |

**Code:**

- `model/nn/layers.py`: `unit_rows`, `UnitRows`, `UnitRowLinear`, `UnitRowConv2d`,
  `SwiGLU(unit_rows=)`, and `affine=` on RMSNorm, LayerNorm and GroupNorm32.
- `model/nn/blocks.py`: the flag on the blocks. `init_writers("output")` scales the writer's gain
  instead of its weight, which gives the same initial write.
- `checks/unit_rows.py`:
  - rows rescaled 0.1–10× leave the block output unchanged (< 1e-6, fp64);
  - gradients are tangent (radial < 1e-8);
  - only the writer has a gain;
  - the writer init lands on the gain;
  - both block types train.
- `torch.compile`: no graph breaks, and compiled matches eager to about 1e-6.

**Checkpoints.** Flag-on and flag-off checkpoints don't load into each other, because the flag adds
gains and drops the norm weights.

**Commits:** `730865c` (head `preserve_init`), `fb3db9c` (unit-row blocks), `b21cdf8` (the toy).

**What to expect at LD's learning rate (1e-4).** The raw norms barely grow, so the main visible
effect should be that the `encoder_output/` and `output_latent/` magnitudes stop climbing. The extra
headroom only matters if the learning rate goes up.

---

## 11. Open items

- [ ] A real run with `unit_rows` on (check the GPUs first).
- [ ] Projection / "both" with a decaying schedule (inverse-sqrt), for a fair comparison with
  normalize-at-use.
- [ ] The A/B/C/D owner comparison on the LM toy and on a regression (denoising AE) toy.
- [ ] A one-sided QK gain, and a gain on the gate branch.
- [ ] An input whose amplitude carries information: per-sample input scaling, with no input norm.
- [ ] The static degree tracer plus the scale-Jacobian spectrum (`free_scales.py`).
- [ ] The weight-decay baseline, Peri-LN, and a second seed.
- [ ] The fused epilogue-scale kernel, if the torch version turns out to be slow.
