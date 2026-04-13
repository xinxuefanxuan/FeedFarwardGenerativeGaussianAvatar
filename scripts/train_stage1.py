"""Entry script for Stage 1 MVP forward/training skeleton."""

from __future__ import annotations

import argparse

import torch

from models.stage1_prior.stage1_pipeline import Stage1CanonicalPrior
from trainers.stage1_trainer import Stage1Trainer
from utils.config import load_yaml


def _dummy_mesh_batch(batch_size: int, num_views: int, h: int, w: int, uv_resolution: int) -> dict[str, torch.Tensor | str]:
    yy, xx = torch.meshgrid(torch.linspace(-0.2, 0.2, 32), torch.linspace(-0.2, 0.2, 32), indexing="ij")
    zz = torch.ones_like(xx)
    verts = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

    faces = []
    for y in range(31):
        for x in range(31):
            i = y * 32 + x
            faces.append([i, i + 1, i + 32])
            faces.append([i + 1, i + 33, i + 32])
    faces = torch.tensor(faces, dtype=torch.long)

    intr = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    intr[..., 0, 0] = 500.0
    intr[..., 1, 1] = 500.0
    intr[..., 0, 2] = w / 2.0
    intr[..., 1, 2] = h / 2.0

    tfm = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)

    uv_verts = torch.stack([
        (verts[:, 0] - verts[:, 0].min()) / (verts[:, 0].max() - verts[:, 0].min() + 1e-8),
        (verts[:, 1] - verts[:, 1].min()) / (verts[:, 1].max() - verts[:, 1].min() + 1e-8),
    ], dim=-1)

    return {
        "images": torch.rand(batch_size, num_views, 3, h, w),
        "confidence": torch.rand(batch_size, num_views, 1, uv_resolution, uv_resolution),
        "mesh_vertices": verts.unsqueeze(0).repeat(batch_size, 1, 1),
        "mesh_faces": faces,
        "uv_vertices": uv_verts,
        "uv_faces": faces,
        "intrinsics": intr,
        "transform_matrices": tfm,
        "template_mesh_path": "/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _ = cfg

    model = Stage1CanonicalPrior()
    trainer = Stage1Trainer(model)

    dummy = _dummy_mesh_batch(batch_size=1, num_views=2, h=256, w=256, uv_resolution=256)
    out = trainer.training_step(dummy)
    print("Stage1 skeleton run OK:", out.keys())


if __name__ == "__main__":
    main()
