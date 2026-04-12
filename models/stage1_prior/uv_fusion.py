"""Canonical UV mapping and fusion placeholders."""

from __future__ import annotations

import torch
from torch import nn


class UVFusionModule(nn.Module):
    """Fuse multi-view projected features in canonical UV space.

    TODO: confirm fusion strategy and confidence weighting definition.
    """

    def forward(self, uv_features: torch.Tensor) -> torch.Tensor:
        return uv_features.mean(dim=1) if uv_features.ndim >= 2 else uv_features
