"""Geometry map builders for Stage 1 canonical prior MVP."""

from __future__ import annotations

from typing import Dict

import torch


def build_uv_valid_mask(confidence_map: torch.Tensor, threshold: float = 1.0e-6) -> torch.Tensor:
    if confidence_map.ndim != 4 or confidence_map.shape[1] != 1:
        raise ValueError(f"Expected confidence_map shape [B,1,H,W], got {tuple(confidence_map.shape)}")
    return (confidence_map > threshold).to(confidence_map.dtype)


def build_uv_position_map(
    fused_uv_features: torch.Tensor,
    uv_valid_mask: torch.Tensor,
    canonical_vertices: torch.Tensor | None = None,
) -> torch.Tensor:
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
        mean_xyz = canonical_vertices.mean(dim=1).view(b, 3, 1, 1)
        pos = pos + mean_xyz

    return pos * uv_valid_mask


def build_uv_normal_map(uv_position_map: torch.Tensor, uv_valid_mask: torch.Tensor) -> torch.Tensor:
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
    uv_valid_mask: torch.Tensor | None = None,
    uv_position_map: torch.Tensor | None = None,
    uv_normal_map: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Build geometry maps; prefer projection-derived atlas maps when provided."""
    if uv_valid_mask is None:
        uv_valid_mask = build_uv_valid_mask(fused_confidence)
    if uv_position_map is None:
        uv_position_map = build_uv_position_map(fused_uv_features, uv_valid_mask, canonical_vertices=canonical_vertices)
    if uv_normal_map is None:
        uv_normal_map = build_uv_normal_map(uv_position_map, uv_valid_mask)

    return {
        "uv_valid_mask": uv_valid_mask,
        "uv_position_map": uv_position_map,
        "uv_normal_map": uv_normal_map,
    }
