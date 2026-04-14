"""Canonical UV fusion module for Stage 1 MVP."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class UVFusionModule(nn.Module):
    """Fuse multi-view UV features with visibility/confidence-aware weights."""

    def forward(
        self,
        uv_features: torch.Tensor,
        visibility: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if uv_features.ndim != 5:
            raise ValueError(f"Expected uv_features [B,V,C,H,W], got {tuple(uv_features.shape)}")

        b, v, _, h, w = uv_features.shape
        device = uv_features.device
        dtype = uv_features.dtype

        if visibility is None:
            visibility = torch.ones((b, v, 1, h, w), device=device, dtype=dtype)
        if confidence is None:
            confidence = torch.ones((b, v, 1, h, w), device=device, dtype=dtype)

        raw = torch.clamp(visibility, min=0.0) * torch.clamp(confidence, min=0.0)

        if v == 1:
            weights = (raw > 0).to(dtype)
            fused = uv_features[:, 0] * raw[:, 0]
            fused_confidence = raw[:, 0].clamp(max=1.0)
            return {
                "fused_uv_features": fused,
                "fused_confidence": fused_confidence,
                "fusion_weights": weights,
            }

        denom = raw.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        weights = raw / denom
        fused = (uv_features * weights).sum(dim=1)
        fused_confidence = raw.sum(dim=1).clamp(max=1.0)

        return {
            "fused_uv_features": fused,
            "fused_confidence": fused_confidence,
            "fusion_weights": weights,
        }
