"""Entry script for Stage 2 MVP debug training."""

from __future__ import annotations

import argparse

import torch

from models.stage2_gaussian.stage2_pipeline import Stage2GaussianAvatar
from trainers.stage2_trainer import Stage2Trainer
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _ = cfg

    model = Stage2GaussianAvatar(uv_feature_dim=32, color_dim=16)
    trainer = Stage2Trainer(model)
    b, v, huv, wuv, himg, wimg, cuv = 1, 3, 64, 64, 128, 128, 32
    uv_valid_mask = (torch.rand(b, 1, huv, wuv) > 0.3).float()
    batch = {
        "uv_valid_mask": uv_valid_mask,
        "uv_position_map": torch.randn(b, 3, huv, wuv) * uv_valid_mask,
        "uv_normal_map": torch.nn.functional.normalize(torch.randn(b, 3, huv, wuv), dim=1) * uv_valid_mask,
        "uv_feature_map": torch.randn(b, cuv, huv, wuv) * uv_valid_mask,
        "uv_confidence_map": torch.rand(b, 1, huv, wuv) * uv_valid_mask,
        "target_images": torch.rand(b, v, 3, himg, wimg),
        "target_masks": (torch.rand(b, v, 1, himg, wimg) > 0.5).float(),
        "intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1),
        "extrinsics": torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1),
    }
    batch["intrinsics"][..., 0, 0] = wimg * 0.8
    batch["intrinsics"][..., 1, 1] = himg * 0.8
    batch["intrinsics"][..., 0, 2] = wimg / 2.0
    batch["intrinsics"][..., 1, 2] = himg / 2.0
    batch["extrinsics"][..., 2, 3] = 2.5

    out = trainer.training_step(batch)
    print("Stage2 MVP run OK:", out.keys())
    print("Loss:", float(out["loss"].detach().cpu()))


if __name__ == "__main__":
    main()
