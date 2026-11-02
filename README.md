# LatentDragon

Work in progress.

Image and video autoencoders (LatentMaid), a vector quantizer for them (KohakuVQ), and the code to
train both.

- **LatentMaid** is an autoencoder family. Its encoder works on pixel-unshuffled features with
  cheap blocks: a depthwise 3x3 convolution and a SwiGLU MLP. An SD-VAE style convolutional encoder
  and decoder sit next to it. The bottleneck is continuous (identity or KL, optionally nested) or
  discrete (KohakuVQ). Video uses the same blocks with causal temporal convolution.
- **KohakuVQ** generates a large codebook from a low-dimensional learned base with a small shared MLP.
  Every code update moves the shared weights, and with them every code, and the
  base costs `N x d` parameters instead of `N x D`. SimVQ is its linear special case.
- **Training** runs on its own loop with GAN schemes, EMA, exact resume and staged
  freezing. Fused Triton kernels for the VQ hot path are included as an add-on.

## LatentMaid

An autoencoder is `encoder -> bottleneck -> trunk -> head`. Each part is registered in a registry
and built from the `ARCH` dict of a config, so a config picks parts instead of editing code. The decoder is a trunk plus
a head (its last projection); a stage-two run can freeze everything but the head.

**MLP encoder and decoder.** A stage is `(scale, hidden, mlp, blocks)`: pixel-unshuffle by
`scale`, a 1x1 projection to `hidden` channels, then `blocks` blocks. A block is

```text
x + SwiGLU(DepthwiseConv3x3(RMSNorm(x)))
```

with the SwiGLU applied over channels at every position (`mlp` hidden units). The decoder mirrors
the encoder with pixel-shuffle, and ends in a 1x1 convolution to image channels. Attention can be
inserted per stage. The MLP blocks are local operations, so every output position depends on a
bounded receptive field.

**Convolutional encoder and decoder.** SD-VAE style: residual blocks (norm, SiLU, 3x3 convolution,
twice) with a stride-2 convolution down and nearest-neighbour upsampling plus convolution up.
Norms are `rms`, `layer` or `group32`.

**Video.** The MLP blocks with a causal temporal convolution mixing frames, temporal down- and
upsampling, and the first frame as a keyframe. `forward_stream` encodes or decodes a clip in chunks
with cached frames and gives the same result as the full forward.

**Bottlenecks.**

| Name | What it does |
| --- | --- |
| `identity` | the encoder output is the latent |
| `kl` | the encoder emits mean and log-variance, sampled in training; KL summed over channels and positions, averaged over the batch |
| `nested` | keeps a prefix of the latent channels per sample, cycling through the widths, for example `(8, 16, 32, 64)`; validation reports every width. Around `vq`, `unit="partition"` counts partitions and `jitter` shifts a sample's width by up to one step |
| `vq` | KohakuVQ, below |

**Architecture in the config.** `ARCH` is a dict of `AEArchConfig` fields written in the config
file, so every encoder, bottleneck, quantizer and trunk setting is a knob. `configs/ae/arch/` holds
the published layouts, pulled in with `use_config`:

| File | Encoder / decoder | Latent | Bottleneck |
| --- | --- | --- | --- |
| `sdvae.py` | SD-VAE conv / SD-VAE conv | f8, 4 ch | KL |
| `fluxvae.py` | SD-VAE conv / SD-VAE conv | f8, 16 ch | KL |
| `latentmaid/2222.py` | MLP, 2 blocks at f1, f2, f4, f8 / SD-VAE conv | f8, 4 ch | KL |
| `latentmaid/0222.py` | MLP, 2 blocks at f2, f4, f8 / SD-VAE conv | f8, 4 ch | KL |
| `latentmaid/0022.py` | MLP, 2 blocks at f4, f8 / SD-VAE conv | f8, 4 ch | KL |

## KohakuVQ

A vector quantizer assigns an encoder output `z` to its nearest code `c_k`, and the straight-through
estimator passes the gradient to the encoder. The codebook only learns from the codes that were
selected, so in a large codebook most codes never move and drift away from the data.

SimVQ (Zhu et al., 2024) writes the codebook as `c_k = b_k W`: a frozen random base `B` and one
learned linear layer `W`. A step on `W` moves every code at once. It has three limits:

- With a base of dimension `d < D` the codes span at most `d` dimensions, so the base has to be full
  rank, `N x D`.
- A linear map of a Gaussian base is Gaussian, so the codes form one ellipsoid.
- The base must stay frozen; training it brings the collapse back.

KohakuVQ replaces the linear layer with a small MLP over a low-dimensional base:

$$c_k = s \odot f_\theta(b_k), \qquad f_\theta(b) = W_2\,\sigma(W_1 b + \beta_1) + \beta_2$$

- `B` is `N x d` with `d` around 8 to 16, and it can be trained.
- `f_theta` is shared by all codes, with SiLU, GELU or Mish.
- `s` is a per-level scalar and an optional per-dimension scale, both initialised to one.
- SimVQ is the case `d = D`, `f` linear, `B` frozen.

A nonlinear projector can reach full rank and non-Gaussian code distributions from a `d`-dimensional base, and
that base leaves each code little freedom to move on its own; that is the intended reason the base can
be trained. The codebook costs `N d + d h + h D` parameters. A 65536-code, 256-channel codebook has about 1.6 M
instead of 16.8 M.

The quantizers are `kohaku` and `naive`, both with straight-through backward, and the converter
reads checkpoints of those two.

**Multiple codebooks.** A latent of `C` channels splits into `P` partitions, and each partition
descends `R` residual levels. Both are one grid, so one code path covers plain VQ, partitioned VQ,
RVQ and the product of the two.

| `partition_mode` | Base | Projector |
| --- | --- | --- |
| `shared` | one | one per level, used by every partition |
| `indep` | one per partition | one per partition and level |
| `hyper` | one | one per level that generates the codebooks of all partitions at once (HyperCodebook); with `separated_proj`, one per partition and level |

**Loss.** `vq_loss = beta * ||sg[c] - z||^2 + ||c - sg[z]||^2`. The optional inverse VQ loss pulls
every code towards its nearest encoder output in the batch, so codes nobody picked still drift
towards the data: `ivq_loss_weight * mean((c - nearest_z)^2)`.

**Init.** The base std is `D^-1/2`, doubled for an MLP projector, so codes start near unit norm.
Projector weights use `fan_in^-1/2` and biases start at zero.

**Two quantizers share the grid.** `kohaku` is the projected codebook above. `naive` is a plain
learned table with the same partitions and levels.

A minimal use:

```python
from latentdragon.vq import QUANTIZER

vq = QUANTIZER.get("kohaku")(
    emb_dim=256, n_embed=65536, base_dim=16, mlp_projector=True, mlp_hidden_dim=2048,
    trainable_base=True, dim_scale=True, partition_mode="hyper", ivq_loss_weight=1e-3,
)
z_q, info = vq(z)    # z: (..., 256), channels last
```

`info` carries the indices, `vq_loss`, code usage and perplexity.

### Kernels

`latentdragon_kernels` holds the VQ hot path and the 16-bit optimizer, each op with an eager
PyTorch reference:

| Op | What it does |
| --- | --- |
| `quantize` | nearest-code search and gather in one Triton pass; the kernel tiles over the codebook and keeps a running minimum (its distance dots run in tf32 for fp32 input) |
| `nearest` | the same search without gradients, for the inverse VQ loss |
| `mlp_project` | the two-layer projector forward and backward; the `gemm` backend computes it on `bmm` in hidden chunks, so no `(G, N, h)` hidden tensor is held between forward and backward |
| `adam16` | Adam with 16-bit state and fp32 registers, the second moment stored as its root, and stochastic rounding on writeback |

An op is picked once when the model is built (`resolve`), by device and card family. `gemm` serves
sm120 and the Triton kernels sm80 or newer; everything else runs the eager path. Block sizes are
timed per card and stored in a table. Precision and throughput are in
[`docs/performance`](docs/performance/README.md).

## Install

Python 3.13, PyTorch 2.13 or newer.

```bash
git clone https://github.com/KohakuBlueleaf/LatentDragon && cd LatentDragon
uv venv -p 3.13 && source .venv/bin/activate
uv pip install -e ".[kernels,wandb]"
```

Extras: `kernels` (Triton), `video` (`av`), `wandb`, `export` (diffusers), `dev` (ruff, black).
The perceptual losses come from [SenseCraft](https://github.com/KohakuBlueleaf/SenseCraft) and
runs are launched with [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) (`kogine`).

## Data

An `image_folder` source reads every `.jpg`, `.jpeg`, `.png` and `.webp` image under a root, recursively. Grayscale, palette and RGBA
files are converted to RGB. The first run writes the file list to `imgonly_metadata_filtered.npy` in
that root, so the folder must be writable once; later runs read that file and do not rescan. A `video_folder` source does the same for videos.

The recipes read `LATENTDRAGON_DATA`, a folder with `imagenet/train` and `imagenet/val`.
`configs/ae/data/multi.py` reads several roots from `LATENTDRAGON_TRAIN_ROOTS`, separated by the OS
path separator.

## Train

```bash
export LATENTDRAGON_DATA=/path/to/data
kogine run scripts/train/ae.py --config configs/ae/recipes/latentmaid/0222.py
```

Every knob is an UPPER_CASE global in `scripts/train/ae.py`, and `--set NAME=value` overrides one:

```bash
kogine run scripts/train/ae.py --config configs/ae/recipes/latentmaid/0222.py \
  --set LR=1e-4 --set EPOCHS=10
```

The recipes use four GPUs, and `BATCH_SIZE` is per GPU. For other hardware write a config that
builds on a recipe, for example `my_configs/one_gpu.py`:

```python
from kohakuengine import use_config

use_config("../configs/ae/recipes/latentmaid/0222.py")  # relative to this file
DEVICES = [0]
BATCH_SIZE = 16
```

Every knob problem is reported together, before any GPU work:

```text
3 config problem(s):
  - unknown knob BATCH_SZE; did you mean BATCH_SIZE?
  - GAN_SCHEME r3gan needs CRITIC
  - set exactly one of MAX_STEPS and EPOCHS
```

**What a run does.**

- Loss: sensecraft terms (`LOSS_CONFIGS`, a list of `sensecraft.Term`) on the reconstruction, plus the bottleneck loss.
- GAN: `taming` (hinge, with the batch-level adaptive weight at the head's last layer), `r3gan` or
  `stylegan2`, with a PatchGAN critic. It starts at `disc_start`.
- Recipes train in fp16 with fp32 weights and a loss scaler per network. `bf16`, `fp32`, `bf16-true` and `fp16-true` are also available; the last two hold 16-bit weights and need `OPTIMIZER=adam16`.
- Weight EMA, used for validation and export. Validation reports rFID, PSNR, SSIM and, for VQ,
  code usage and perplexity.
- Each network has its own optimizer, schedule, gradient scaler and clipping; the gradient is
  averaged across GPUs once per step.

**Run folder.** `runs/<name>/<id>/` holds `config.json`, `git.json`, `metrics.jsonl`,
`model_summary.txt`, `timing.json` (time per phase), `samples/` and `checkpoints/step_XXXXXXXX/`. Checkpoints are
written atomically and hold the weights, optimizer state, EMA, RNG state per GPU and the data
position.

**Logging.** Every step is logged, every metric on every step (`METRICS_EVERY=1`); rows are copied
to the host in batches of `FLUSH_EVERY`. Rank 0 writes each row to `metrics.jsonl`; an interactive terminal shows a
progress bar with the headline metrics, and a log file gets one line per flush. With
`WANDB_PROJECT` set, the same rows go to wandb as Lightning's `WandbLogger` sends them: the run is
named `NAME`, every row carries `trainer/global_step` and `epoch`, panels plot against
`trainer/global_step`, and the script, `config.json` and `git.json` are saved to the run.
`WANDB_OFFLINE` keeps the run local, `WANDB_RUN_ID` with `WANDB_RESUME` continues one, and
`WANDB_WATCH` logs gradient histograms every N steps.

**Module summary.** Before the first step rank 0 prints a Lightning-style table of the model,
critic, latent terms and loss networks down to `SUMMARY_DEPTH` (2 by default, Lightning uses 1):
parameters, trainable parameters and train/eval mode per module, then the trainable, frozen and
total counts and their size in MB. The table is kept as `model_summary.txt`; wandb gets the file and
the totals as `model/params_*` summary values.

| Section | What it holds |
| --- | --- |
| `train/` | `eloss`, `psnr` and `ssim` smoothed, `nll`, `loss`, each loss term, `kl`, `z_mean`, `z_std` |
| `vq/` | smoothed `loss`, `usage`, `perplexity`, `perplexity_rate`, the loss parts, `usage_ema` |
| `gan/` | `g_loss`, `d_loss`, critic logits, the adaptive weight and its scale |
| `<key>_ema` | next to every raw scalar of `train/`, `vq/`, `gan/` and `optim/grad_norm/`: our own EMA, decay `min(0.995, step / (step + 10))`, so a raw spike and the trend sit in one section |
| `latent_stat_mean/`, `latent_stat_std/` | per latent channel |
| `optim/` | learning rate per group, gradient norm, skipped steps and loss scale per network |
| `raw/` | the unsmoothed values behind the smoothed ones |
| `perf/` | step time and samples per second |
| `val/` | rFID, PSNR, SSIM, code usage and perplexity on the validation set |
| `train/samples` | real images, the latent as RGB, the reconstruction, one per nested level |

**Resume.** `--set RESUME=runs/<name>/<id>` continues the run. A resume compares the knobs with
the run's `config.json`; knobs that may differ go in `RESUME_IGNORE`.

**Stages.** Stage two starts from a stage-one checkpoint and freezes parts of the model. The
tokens stay the same, so a generator trained on them stays valid:

```python
INIT_FROM = "runs/<name>/<id>/checkpoints/step_00400000"
FREEZE = ["encoder", "bottleneck", "trunk"]
```

`FREEZE` takes any dotted model path. Frozen parts stay in eval mode and out of every optimizer.

## Recipes

Every recipe in `configs/ae/recipes/` is ImageNet, fp16 and the taming setup in
`configs/ae/train/taming.py` (MSE + 0.5 LPIPS-VGG, KL 1e-6, hinge BatchNorm PatchGAN from step 25000
with the adaptive weight) on one arch file of the same name:

| Recipe | Arch |
| --- | --- |
| `sdvae.py` | `arch/sdvae.py` |
| `fluxvae.py` | `arch/fluxvae.py` |
| `latentmaid/2222.py`, `0222.py`, `0022.py` | `arch/latentmaid/` of the same name |

## Convert and export

LatentMaid `.ckpt` and safetensors checkpoints convert to a LatentDragon export:

```bash
kogine run scripts/tools/convert_legacy.py --set SOURCE=old.ckpt --set TARGET=converted
```

`scripts/export/export_run.py` writes a checkpoint as `FORM=native` (`model.safetensors` and
`spec.json`), `diffusers` (a folder for `AutoModel.from_pretrained(trust_remote_code=True)`),
`legacy_safetensors` or `legacy_ckpt`. A `.ckpt` carries pickled code; load files you trust.

## Decode large latents

`decode` runs a latent through a strategy selected by name. Tile sizes are output pixels.

```python
from latentdragon.inference import decode

out = decode(model, z, {"name": "tiled", "tile": [512, 512]})
out = decode(model, z, {"name": "staged_tiled", "stages": 2, "tile": [512, 512]})
out = decode(model, z, "streaming")
out = decode(model, z, {"name": "streaming_tiled", "tile": [1024, 1024]})
```

| Strategy | Use it for |
| --- | --- |
| `tiled` | large images and video; `tile_t` also tiles time, and `inner` chooses how each tile decodes |
| `staged_tiled` | running the early trunk stages whole and tiling the late ones |
| `streaming` | long video at constant memory, frame chunks as they finish |
| `streaming_tiled` | long, large video, one causal cache per tile |

Overlap comes from `model.reach()`, which is summed from what each block declares, so a tiled decode
matches the full decode up to floating point. Attention and group norm read every position; on those
models set `overlap` to accept an approximate result. Streaming needs the video trunk. See
[inference](docs/concepts/inference.md).

## Extend

Parts are registered by name, and a dotted path works wherever a name does.

| Add | Registry |
| --- | --- |
| encoder, trunk, block, norm | `ENCODER`, `TRUNK`, `BLOCK`, `NORM` |
| head and its objective | `HEAD`, `HEAD_OBJECTIVES` |
| bottleneck, codebook | `BOTTLENECK`, `CODEBOOK`, `QUANTIZER` |
| critic, GAN scheme | `CRITIC`, `GAN_SCHEME` |
| decode strategy | `DECODE` |
| latent augmentation, loss, frozen teacher | `LATENT_VIEW`, `LATENT_TERM`, `TEACHER` |
| a new kind of training | a `Task` |

## Layout

```text
src/latentdragon_common/    registry, config checks, logging
src/latentdragon_kernels/   Triton and eager ops, per-card tuning
src/latentdragon_data/      image and video sources, resumable loader
src/latentdragon/
  model/  bottleneck/  vq/  domain/  gan/  latent/  objectives/
  tasks/  training/  eval/  inference/  io/
configs/ae/                 arch, data, precision, train setups, recipes
scripts/                    train, export, tools
experiments/                decode strategies, kernel precision, throughput
docs/                       concepts, guides
checks/                     behavior scripts, run by hand; see checks/README.md
```

Start at [docs/README.md](docs/README.md). Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR;
feature PRs need an approved issue first.

## References

- Zhu et al., *Addressing Representation Collapse in Vector Quantized Models with One Linear Layer*,
  2024 (SimVQ). [arXiv:2411.02038](https://arxiv.org/abs/2411.02038)

## License

[Apache-2.0](LICENSE)
