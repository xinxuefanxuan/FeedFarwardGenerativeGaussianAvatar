"""Bridge minimal Gaussian params to view-space render tensors."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class Stage2RenderBridge(nn.Module):
    """Very small splat renderer bridge for debug/training plumbing."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _project(
        xyz: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = xyz.shape[0]
        pts_h = torch.cat([xyz, torch.ones((n, 1), device=xyz.device, dtype=xyz.dtype)], dim=-1)
        cam = (extrinsics @ pts_h.T).T[:, :3]
        z = cam[:, 2]
        z_safe = torch.clamp(z, min=1.0e-6)
        uv_h = (intrinsics @ torch.stack([cam[:, 0] / z_safe, cam[:, 1] / z_safe, torch.ones_like(z_safe)], dim=-1).T).T
        return uv_h[:, :2], z

    def forward(
        self,
        gaussians: Dict[str, torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        b, v = intrinsics.shape[:2]
        h, w = image_size
        device = intrinsics.device
        dtype = intrinsics.dtype

        rgb = torch.zeros((b, v, 3, h, w), device=device, dtype=dtype)
        alpha = torch.zeros((b, v, 1, h, w), device=device, dtype=dtype)

        xyz = gaussians["gaussian_xyz"]
        opacity = gaussians["gaussian_opacity"]
        color_feat = gaussians["gaussian_color_feat"]
        color = torch.sigmoid(color_feat[..., :3])

        for bi in range(b):
            for vi in range(v):
                uv, z = self._project(xyz[bi], intrinsics[bi, vi], extrinsics[bi, vi])
                x = torch.round(uv[:, 0]).to(torch.long)
                y = torch.round(uv[:, 1]).to(torch.long)
                valid = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
                if not valid.any():
                    continue
                xv = x[valid]
                yv = y[valid]
                av = opacity[bi, valid, 0].clamp(0.0, 1.0)
                cv = color[bi, valid]

                alpha[bi, vi, 0, yv, xv] = torch.clamp(alpha[bi, vi, 0, yv, xv] + av, 0.0, 1.0)
                for c in range(3):
                    rgb[bi, vi, c, yv, xv] = rgb[bi, vi, c, yv, xv] * (1.0 - av) + cv[:, c] * av

        return {"rendered_images": rgb, "rendered_alpha": alpha}
