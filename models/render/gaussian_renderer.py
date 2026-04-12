"""Differentiable Gaussian renderer placeholder."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GaussianRenderer(nn.Module):
    """Render Gaussian attributes into an image-like tensor.

    TODO: plug in actual differentiable Gaussian rasterizer backend.
    """

    def forward(self, gaussian_attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent = gaussian_attributes["latent"]
        return latent
