"""Stage 2 feedforward Gaussian avatar pipeline skeleton."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.render.gaussian_renderer import GaussianRenderer
from models.stage2_gaussian.anchor_init import initialize_gaussian_anchors
from models.stage2_gaussian.gaussian_decoder import GaussianAttributeDecoder


class Stage2GaussianAvatar(nn.Module):
    """Generate Gaussian avatar from canonical representations."""

    def __init__(self, num_anchors: int = 20000, in_dim: int = 32) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.decoder = GaussianAttributeDecoder(in_dim=in_dim)
        self.renderer = GaussianRenderer()

    def forward(self, stage1_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        uv = stage1_outputs["canonical_uv"]
        anchors = initialize_gaussian_anchors(uv, num_anchors=self.num_anchors)
        gaussian_attrs = self.decoder(anchors)
        render = self.renderer(gaussian_attrs)
        return {"gaussians": gaussian_attrs, "render": render}
