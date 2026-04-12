"""Geometry map builders for Stage 1 canonical prior MVP."""

from __future__ import annotations

from typing import Dict

import torch


def build_uv_valid_mask(confidence_map: torch.Tensor, threshold: float = 1.0e-6) -> torch.Tensor:
    """Build UV valid mask from fused confidence.

    Args:
        confidence_map: Tensor [B, 1, H, W].
    """
    if confidence_map.ndim != 4 or confidence_map.shape[1] != 1:
        raise ValueError(f"Expected confidence_map shape [B,1,H,W], got {tuple(confidence_map.shape)}")
    return (confidence_map > threshold).to(confidence_map.dtype)


def build_uv_position_map(
    fused_uv_features: torch.Tensor,
    uv_valid_mask: torch.Tensor,
    canonical_vertices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a coarse UV position map.

    MVP behavior:
    - If canonical vertices provided, use their mean xyz as global offset.
    - Use normalized UV grid as xy scaffold.
    - Gate by uv_valid_mask.
    """
    if fused_uv_features.ndim != 4:
        raise ValueError(f"Expected fused_uv_features shape [B,C,H,W], got {tuple(fused_uv_features.shape)}")

    b, _, h, w = fused_uv_features.shape
    device = fused_uv_features.device
    dtype = fused_uv_features.dtype

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    pos = torch.stack([xx, yy, torch.zeros_like(xx)], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)

    if canonical_vertices is not None:
        if canonical_vertices.ndim != 3 or canonical_vertices.shape[-1] != 3:
            raise ValueError("canonical_vertices must be [B,N,3]")
        mean_xyz = canonical_vertices.mean(dim=1).view(b, 3, 1, 1)
        pos = pos + mean_xyz

    return pos * uv_valid_mask


def build_uv_normal_map(uv_position_map: torch.Tensor, uv_valid_mask: torch.Tensor) -> torch.Tensor:
    """Build normal map from UV position finite differences."""
    if uv_position_map.ndim != 4 or uv_position_map.shape[1] != 3:
        raise ValueError(f"Expected uv_position_map shape [B,3,H,W], got {tuple(uv_position_map.shape)}")

    dx = uv_position_map[:, :, :, 1:] - uv_position_map[:, :, :, :-1]
    dy = uv_position_map[:, :, 1:, :] - uv_position_map[:, :, :-1, :]

    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))

    normals = torch.cross(dx, dy, dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-8)
    return normals * uv_valid_mask


def build_geometry_maps(
    fused_uv_features: torch.Tensor,
    fused_confidence: torch.Tensor,
    canonical_vertices: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper to build valid/position/normal maps."""
    uv_valid_mask = build_uv_valid_mask(fused_confidence)
    uv_position_map = build_uv_position_map(fused_uv_features, uv_valid_mask, canonical_vertices=canonical_vertices)
    uv_normal_map = build_uv_normal_map(uv_position_map, uv_valid_mask)
    return {
        "uv_valid_mask": uv_valid_mask,
        "uv_position_map": uv_position_map,
        "uv_normal_map": uv_normal_map,
    }
