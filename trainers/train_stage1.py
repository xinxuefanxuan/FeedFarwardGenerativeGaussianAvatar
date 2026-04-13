"""Debug forward entry for Stage 1 MVP chain."""

from __future__ import annotations

import argparse

import torch

from models.stage1_prior.stage1_pipeline import Stage1CanonicalPrior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--uv-resolution", type=int, default=256)
    args = parser.parse_args()

    model = Stage1CanonicalPrior(uv_resolution=args.uv_resolution)

    images = torch.rand(args.batch_size, args.num_views, 3, args.height, args.width)
    confidence = torch.rand(args.batch_size, args.num_views, 1, args.uv_resolution, args.uv_resolution)
    visibility = (confidence > 0.2).float()

    with torch.no_grad():
        outputs = model(
            {
                "images": images,
                "confidence": confidence,
                "visibility": visibility,
            }
        )

    print("Stage1 MVP forward OK")
    print("fused_uv_feature_map:", tuple(outputs["fused_uv_feature_map"].shape))
    print("uv_valid_mask:", tuple(outputs["uv_valid_mask"].shape), "valid_ratio=", float(outputs["uv_valid_mask"].mean()))
    print("uv_position_map:", tuple(outputs["uv_position_map"].shape))
    print("uv_normal_map:", tuple(outputs["uv_normal_map"].shape))


if __name__ == "__main__":
    main()
