"""Decode per-anchor Gaussian attributes (skeleton)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GaussianAttributeDecoder(nn.Module):
    """Decode Gaussian properties from anchor features.

    TODO: confirm final attribute set (scale/rotation/opacity/color/feature).
    """

    def __init__(self, in_dim: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, anchors: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.net(anchors)
        return {"latent": latent}
