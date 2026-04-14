"""Minimal Stage-2 debug forward/render/loss script."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.stage2_gaussian.stage2_pipeline import Stage2GaussianAvatar
from trainers.stage2_trainer import Stage2Trainer


def _save_rgb(x_chw: torch.Tensor, path: Path) -> None:
    x = x_chw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    img = (np.transpose(x, (1, 2, 0)) * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def _save_gray(x_hw: torch.Tensor, path: Path) -> None:
    x = x_hw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    img = (x * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


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
    parser.add_argument("--out-dir", type=str, default="outputs/stage2_debug")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = outputs["rendered_images"][0]      # [V,3,H,W]
    alpha = outputs["rendered_alpha"][0]          # [V,1,H,W]
    target = batch["target_images"][0]            # [V,3,H,W]
    num_views = rendered.shape[0]

    _save_rgb(rendered[0], out_dir / "rendered_view0.png")
    _save_gray(alpha[0, 0], out_dir / "rendered_alpha_view0.png")
    _save_rgb(target[0], out_dir / "target_view0.png")

    if num_views > 1:
        _save_rgb(rendered[1], out_dir / "rendered_view1.png")
        _save_gray(alpha[1, 0], out_dir / "rendered_alpha_view1.png")
        _save_rgb(target[1], out_dir / "target_view1.png")

    # Optional quick side-by-side grid: [rendered0 | target0 | alpha0]
    r0 = rendered[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    t0 = target[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    a0 = alpha[0, 0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    a0_rgb = np.stack([a0, a0, a0], axis=0)
    grid = np.concatenate([r0, t0, a0_rgb], axis=2)  # CHW, concat width
    _save_rgb(torch.from_numpy(grid), out_dir / "rendered_vs_target_grid.png")
    print(f"Saved debug images to: {out_dir}")


if __name__ == "__main__":
    main()
