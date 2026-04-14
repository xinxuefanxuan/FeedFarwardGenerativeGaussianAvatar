"""Heads for decoding Gaussian attributes from per-UV latent features."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GaussianParamHead(nn.Module):
    """Predict Gaussian params from latent tokens."""

    def __init__(self, latent_dim: int = 128, color_dim: int = 16) -> None:
        super().__init__()
        self.delta_xyz = nn.Linear(latent_dim, 3)
        self.scale = nn.Linear(latent_dim, 3)
        self.rotation = nn.Linear(latent_dim, 4)
        self.opacity = nn.Linear(latent_dim, 1)
        self.color_feat = nn.Linear(latent_dim, color_dim)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "delta_xyz": self.delta_xyz(latent),
            "scale_raw": self.scale(latent),
            "rotation_raw": self.rotation(latent),
            "opacity_raw": self.opacity(latent),
            "color_feat_raw": self.color_feat(latent),
        }
