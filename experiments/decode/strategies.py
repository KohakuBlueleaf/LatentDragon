"""Question: how far is each decode strategy from decoding the whole latent at once?

Decodes random latents through untrained models on the CPU and prints the largest absolute
difference from `full` per strategy, one JSON row each. Read the numbers against the
output scale: floating point noise is the expected result for an exact strategy.

    python experiments/decode/strategies.py
"""

import json

import torch

from latentdragon.inference import decode, stream_encode
from latentdragon.model import build_autoencoder, get_preset

VIDEO = {
    "latent_dim": 4,
    "encoder_blocks": ((2, 2, 16, 16, 1), (2, 2, 16, 16, 1)),
    "trunk_blocks": ((1, 2, 16, 16, 2), (2, 2, 32, 32, 2), (2, 1, 32, 32, 2)),
}
IMAGE_ENCODER = ((2, 16, 16, 1), (2, 16, 16, 1), (2, 16, 16, 1))
CONV_IMAGE = {
    "latent_dim": 4,
    "encoder_blocks": IMAGE_ENCODER,
    "trunk_blocks": ((2, 32, 1), (2, 32, 1), (2, 16, 1)),
}
MLP_IMAGE = {
    "latent_dim": 4,
    "encoder_blocks": IMAGE_ENCODER,
    "trunk": "mlp",
    "trunk_blocks": ((2, 16, 16, 2), (2, 32, 32, 2), (2, 32, 32, 2)),
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


def tiled_output(model, z: torch.Tensor, case) -> tuple[torch.Tensor, int]:
    """Decode with `case` and count the head calls, one per tile or chunk."""
    calls = []
    hook = model.head.register_forward_hook(lambda *_: calls.append(1))
    output = decode(model, z, case)
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

    video = build_autoencoder(get_preset("video_f16t6c32", **VIDEO)).eval()
    print(json.dumps({"video_reach": str(video.reach())}))
    z = torch.randn(1, 4, 9, 40, 40)
    full = decode(video, z)
    for case in VIDEO_CASES:
        output, calls = tiled_output(video, z, case)
        report("video", output, full, case, calls)

    clip = torch.randn(1, 17, 3, 32, 32)
    whole = video.encoder(clip)
    chunks = list(stream_encode(video, clip))
    report("video_encoder", torch.cat(chunks, dim=2), whole, "stream_encode")

    z = torch.randn(1, 4, 40, 40)
    images = (
        ("conv_image", CONV_IMAGE, CONV_IMAGE_CASES),
        ("mlp_image", MLP_IMAGE, MLP_IMAGE_CASES),
    )
    for label, arch, cases in images:
        image = build_autoencoder(get_preset("latentmaid_f8c4", **arch)).eval()
        print(json.dumps({f"{label}_reach": str(image.reach())}))
        full = decode(image, z)
        for case in cases:
            output, calls = tiled_output(image, z, case)
            report(label, output, full, case, calls)


if __name__ == "__main__":
    main()
