"""UV refinement placeholder network."""

from __future__ import annotations

import torch
from torch import nn


class UVRefinementNet(nn.Module):
    """Small placeholder refinement net for canonical UV features."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.refine = nn.Identity()

    def forward(self, uv_tensor: torch.Tensor) -> torch.Tensor:
        return self.refine(uv_tensor)
