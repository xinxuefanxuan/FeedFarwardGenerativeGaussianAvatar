"""Minimal Stage-2 debug forward/render/loss script."""

from __future__ import annotations

import argparse

import torch

from models.stage2_gaussian.stage2_pipeline import Stage2GaussianAvatar
from trainers.stage2_trainer import Stage2Trainer


def _make_dummy_batch(batch_size: int, num_views: int, huv: int, wuv: int, himg: int, wimg: int, cuv: int) -> dict:
    uv_valid_mask = (torch.rand(batch_size, 1, huv, wuv) > 0.3).float()
    uv_position_map = torch.randn(batch_size, 3, huv, wuv) * uv_valid_mask
    uv_normal_map = torch.nn.functional.normalize(torch.randn(batch_size, 3, huv, wuv), dim=1) * uv_valid_mask
    uv_feature_map = torch.randn(batch_size, cuv, huv, wuv) * uv_valid_mask
    uv_confidence_map = torch.rand(batch_size, 1, huv, wuv) * uv_valid_mask

    target_images = torch.rand(batch_size, num_views, 3, himg, wimg)
    target_masks = (torch.rand(batch_size, num_views, 1, himg, wimg) > 0.5).float()

    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, num_views, 1, 1)
    intrinsics[..., 0, 0] = float(wimg) * 0.8
    intrinsics[..., 1, 1] = float(himg) * 0.8
    intrinsics[..., 0, 2] = float(wimg) / 2.0
    intrinsics[..., 1, 2] = float(himg) / 2.0
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_views, 1, 1)
    extrinsics[..., 2, 3] = 2.5

    return {
        "uv_valid_mask": uv_valid_mask,
        "uv_position_map": uv_position_map,
        "uv_normal_map": uv_normal_map,
        "uv_feature_map": uv_feature_map,
        "uv_confidence_map": uv_confidence_map,
        "target_images": target_images,
        "target_masks": target_masks,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--uv-resolution", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--uv-feature-dim", type=int, default=32)
    parser.add_argument("--gaussian-color-dim", type=int, default=16)
    args = parser.parse_args()

    batch = _make_dummy_batch(
        batch_size=args.batch_size,
        num_views=args.num_views,
        huv=args.uv_resolution,
        wuv=args.uv_resolution,
        himg=args.image_size,
        wimg=args.image_size,
        cuv=args.uv_feature_dim,
    )

    model = Stage2GaussianAvatar(uv_feature_dim=args.uv_feature_dim, color_dim=args.gaussian_color_dim)
    trainer = Stage2Trainer(model)
    out = trainer.training_step(batch)
    outputs = out["outputs"]
    losses = out["losses"]

    print("=== Stage2 MVP debug ===")
    print("gaussian_xyz:", tuple(outputs["gaussian_xyz"].shape))
    print("gaussian_scale:", tuple(outputs["gaussian_scale"].shape))
    print("gaussian_rotation:", tuple(outputs["gaussian_rotation"].shape))
    print("gaussian_opacity:", tuple(outputs["gaussian_opacity"].shape))
    print("gaussian_color_feat:", tuple(outputs["gaussian_color_feat"].shape))
    print("gaussian_uv_index:", tuple(outputs["gaussian_uv_index"].shape))
    print("rendered_images:", tuple(outputs["rendered_images"].shape))
    print("rendered_alpha:", tuple(outputs["rendered_alpha"].shape))
    print("loss_total:", float(losses["loss_total"].detach().cpu()))
    print("loss_image_l1:", float(losses["loss_image_l1"].detach().cpu()))
    print("loss_alpha:", float(losses["loss_alpha"].detach().cpu()))
    print("loss_xyz_reg:", float(losses["loss_xyz_reg"].detach().cpu()))
    print("loss_scale_reg:", float(losses["loss_scale_reg"].detach().cpu()))
    print("loss_opacity_reg:", float(losses["loss_opacity_reg"].detach().cpu()))


if __name__ == "__main__":
    main()
