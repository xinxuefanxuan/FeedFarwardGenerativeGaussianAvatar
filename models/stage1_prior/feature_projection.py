"""Project image features to surface/UV spaces for Stage 1 MVP."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _ensure_bvchw(features: torch.Tensor) -> torch.Tensor:
    """Normalize feature tensor to [B,V,C,H,W]."""
    if features.ndim == 4:
        return features.unsqueeze(1)
    if features.ndim == 5:
        return features
    raise ValueError(f"Expected features ndim 4 or 5, got shape {tuple(features.shape)}")


def project_image_features_to_surface(
    image_features: torch.Tensor,
    uv_resolution: int = 256,
) -> torch.Tensor:
    """MVP surface projection proxy.

    Current proxy behavior:
    - Treat encoder feature maps as already image-aligned surface evidence.
    - Resize per-view feature maps to UV resolution.
    """
    feats = _ensure_bvchw(image_features)
    b, v, c, h, w = feats.shape
    flat = feats.reshape(b * v, c, h, w)
    resized = F.interpolate(flat, size=(uv_resolution, uv_resolution), mode="bilinear", align_corners=False)
    return resized.view(b, v, c, uv_resolution, uv_resolution)


def project_surface_to_uv(
    surface_features: torch.Tensor,
    visibility: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """MVP surface->UV mapping.

    Since explicit mesh rasterization-to-UV is not integrated here, this function forwards
    resized surface features as UV features and attaches visibility/confidence maps.
    """
    if surface_features.ndim != 5:
        raise ValueError(f"Expected surface_features [B,V,C,H,W], got {tuple(surface_features.shape)}")

    b, v, _, h, w = surface_features.shape
    device = surface_features.device
    dtype = surface_features.dtype

    if visibility is None:
        visibility = torch.ones((b, v, 1, h, w), device=device, dtype=dtype)
    if confidence is None:
        confidence = torch.ones((b, v, 1, h, w), device=device, dtype=dtype)

    return {
        "uv_features": surface_features,
        "visibility": visibility,
        "confidence": confidence,
    }
