"""Stage 1 canonical prior builder (MVP chain)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.encoders.dinov2_encoder import DINOv2Encoder
from models.stage1_prior.feature_projection import project_image_features_to_surface, project_surface_to_uv
from models.stage1_prior.geometry_maps import build_geometry_maps
from models.stage1_prior.uv_fusion import UVFusionModule
from models.stage1_prior.uv_refinement import UVRefinementNet


class Stage1PriorBuilder(nn.Module):
    """Build canonical prior by chaining encoder -> projection -> UV fusion -> geometry maps."""

    def __init__(self, uv_resolution: int = 256) -> None:
        super().__init__()
        self.uv_resolution = uv_resolution
        self.encoder = DINOv2Encoder(variant="vitb14", freeze=True)
        self.uv_fusion = UVFusionModule()
        self.uv_refinement = UVRefinementNet()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "images" not in batch:
            raise KeyError("Stage1PriorBuilder expects `images` in batch")

        image_features = self.encoder(batch["images"])
        surface_features = project_image_features_to_surface(image_features, uv_resolution=self.uv_resolution)
        uv_pack = project_surface_to_uv(
            surface_features,
            visibility=batch.get("visibility"),
            confidence=batch.get("confidence"),
        )

        fusion_out = self.uv_fusion(
            uv_pack["uv_features"],
            visibility=uv_pack["visibility"],
            confidence=uv_pack["confidence"],
        )

        refined_uv = self.uv_refinement(fusion_out["fused_uv_features"])
        geom = build_geometry_maps(
            fused_uv_features=refined_uv,
            fused_confidence=fusion_out["fused_confidence"],
            canonical_vertices=batch.get("canonical_vertices"),
        )

        return {
            "image_features": image_features,
            "surface_features": surface_features,
            "uv_features": uv_pack["uv_features"],
            "fused_uv_feature_map": refined_uv,
            "fused_confidence_map": fusion_out["fused_confidence"],
            "fusion_weights": fusion_out["fusion_weights"],
            "uv_valid_mask": geom["uv_valid_mask"],
            "uv_position_map": geom["uv_position_map"],
            "uv_normal_map": geom["uv_normal_map"],
            "canonical_uv": refined_uv,
        }
