"""DINOv2 encoder wrapper (skeleton)."""

from __future__ import annotations

import torch
from torch import nn


class DINOv2Encoder(nn.Module):
    """Image encoder wrapper.

    This class intentionally exposes only the interface in skeleton stage.
    """

    def __init__(self, variant: str = "vitb14", freeze: bool = True) -> None:
        super().__init__()
        self.variant = variant
        self.freeze = freeze
        self.backbone = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into feature tensors.

        Args:
            images: Tensor of shape [B, V, C, H, W] or [B, C, H, W].
        """
        return self.backbone(images)
