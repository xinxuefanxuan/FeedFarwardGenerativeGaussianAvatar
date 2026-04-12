"""Stage 1 pipeline skeleton."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.encoders.dinov2_encoder import DINOv2Encoder
from models.stage1_prior.uv_fusion import UVFusionModule
from models.stage1_prior.uv_refinement import UVRefinementNet


class Stage1CanonicalPrior(nn.Module):
    """Build geometry-consistent canonical priors from sparse observations."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DINOv2Encoder()
        self.uv_fusion = UVFusionModule()
        self.uv_refinement = UVRefinementNet()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run Stage 1 skeleton forward pass.

        Expects key `images` in batch for skeleton validation.
        """
        images = batch["images"]
        image_features = self.encoder(images)
        fused_uv = self.uv_fusion(image_features)
        refined_uv = self.uv_refinement(fused_uv)

        return {
            "canonical_uv": refined_uv,
            "uv_position_map": refined_uv,
            "uv_normal_map": refined_uv,
        }
