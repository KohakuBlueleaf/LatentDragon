"""Question: how far is each decode strategy from decoding the whole latent at once?

Decodes random latents through untrained models on the CPU and prints the largest absolute
difference from `full` per strategy, one JSON row each. Read the numbers against the
output scale: floating point noise is the expected result for an exact strategy.

    python experiments/decode/strategies.py
"""

import json

import torch

from latentdragon.inference import decode, stream_encode
from latentdragon.model import build_autoencoder

# Small untrained layouts; strategy error does not depend on width
VIDEO = {
    "domain": "video",
    "encoder": {
        "name": "latentmaid_video",
        "in_channels": 3,
        "out_channels": 4,
        "stages": [[2, 2, 16, 16, 1], [2, 2, 16, 16, 1]],
        "norm": "rms",
    },
    "latent": "identity",
    "decoder": {
        "name": "latentmaid_video",
        "in_channels": 4,
        "stages": [[1, 2, 16, 16, 2], [2, 2, 32, 32, 2], [2, 1, 32, 32, 2]],
        "norm": "rms",
    },
    "head": {
        "name": "conv",
        "in_channels": 4,
        "out_channels": 3,
        "kernel_size": 1,
        "video": True,
    },
}
CONV_IMAGE = {
    "domain": "image",
    "encoder": {
        "name": "latentmaid",
        "in_channels": 3,
        "out_channels": 4,
        "stages": [[2, 16, 16, 1], [2, 16, 16, 1], [2, 16, 16, 1]],
        "norm": "rms",
        "attn": None,
    },
    "latent": "identity",
    "decoder": {
        "name": "sdvae",
        "in_channels": 4,
        "stages": [[2, 32, 1], [2, 32, 1], [2, 16, 1]],
        "norm": "rms",
        "attn": None,
    },
    "head": {
        "name": "conv",
        "in_channels": 16,
        "out_channels": 3,
        "kernel_size": 3,
        "norm": "rms",
        "act": "mish",
    },
}
MLP_IMAGE = {
    "domain": "image",
    "encoder": {
        "name": "latentmaid",
        "in_channels": 3,
        "out_channels": 4,
        "stages": [[2, 16, 16, 1], [2, 16, 16, 1], [2, 16, 16, 1]],
        "norm": "rms",
        "attn": None,
    },
    "latent": "identity",
    "decoder": {
        "name": "latentmaid",
        "in_channels": 4,
        "stages": [[2, 16, 16, 2], [2, 32, 32, 2], [2, 32, 32, 2]],
        "norm": "rms",
        "attn": None,
    },
    "head": {"name": "conv", "in_channels": 4, "out_channels": 3, "kernel_size": 1},
}
VIDEO_CASES = [
    {"name": "tiled", "tile": [96, 96]},
    {"name": "tiled", "tile": [96, 128], "tile_t": 24},
    {"name": "tiled", "tile": [96, 96], "overlap": 2},
    {"name": "tiled", "tile": [10000, 10000], "tile_t": 24, "overlap_t": 1},
    {"name": "staged_tiled", "stages": 2, "tile": [96, 96]},
    {
        "name": "tiled",
        "tile": [128, 128],
        "inner": {"name": "staged_tiled", "stages": 2, "tile": [96, 96]},
    },
    "streaming",
    {"name": "streaming", "first_tokens": 1, "tokens_per_step": 2},
    {"name": "streaming_tiled", "tile": [96, 96]},
]
CONV_IMAGE_CASES = [
    {"name": "tiled", "tile": [128, 128]},
]
MLP_IMAGE_CASES = [
    {"name": "tiled", "tile": [96, 96]},
    {"name": "staged_tiled", "stages": 2, "tile": [64, 64]},
]


def decoded(model, z: torch.Tensor, case="full") -> torch.Tensor:
    """The output tensor of decoding the decoder input `z` with `case`."""
    return decode(model, {"z": z}, case)["x"]


def tiled_output(model, z: torch.Tensor, case) -> tuple[torch.Tensor, int]:
    """Decode with `case` and count the head projections, one per tile or chunk."""
    calls = []
    hook = model.head.last_layer.register_forward_hook(lambda *_: calls.append(1))
    output = decoded(model, z, case)
    hook.remove()
    return output, len(calls)


def report(label, value, reference, case, calls: int = 1) -> None:
    """Print one row comparing an output with its reference."""
    row = {
        "model": label,
        "case": case,
        "head_calls": calls,
        "shape": list(value.shape),
        "max_abs": (value - reference).abs().max().item(),
        "output_max": reference.abs().max().item(),
    }
    print(json.dumps(row))


def main() -> None:
    torch.manual_seed(0)

    video = build_autoencoder(VIDEO).eval()
    print(json.dumps({"video_reach": str(video.reach())}))
    z = torch.randn(1, 4, 9, 40, 40)
    full = decoded(video, z)
    for case in VIDEO_CASES:
        output, calls = tiled_output(video, z, case)
        report("video", output, full, case, calls)

    clip = torch.randn(1, 17, 3, 32, 32)
    whole = video.encoder.tensor(clip)
    chunks = list(stream_encode(video, clip))
    report("video_encoder", torch.cat(chunks, dim=2), whole, "stream_encode")

    z = torch.randn(1, 4, 40, 40)
    images = (
        ("conv_image", CONV_IMAGE, CONV_IMAGE_CASES),
        ("mlp_image", MLP_IMAGE, MLP_IMAGE_CASES),
    )
    for label, arch, cases in images:
        image = build_autoencoder(arch).eval()
        print(json.dumps({f"{label}_reach": str(image.reach())}))
        full = decoded(image, z)
        for case in cases:
            output, calls = tiled_output(image, z, case)
            report(label, output, full, case, calls)


if __name__ == "__main__":
    main()
