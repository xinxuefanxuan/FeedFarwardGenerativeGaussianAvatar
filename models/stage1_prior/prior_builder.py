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
    """Build canonical prior by chaining encoder -> real projection -> UV fusion -> geometry maps."""

    def __init__(
        self,
        uv_resolution: int = 256,
        template_mesh_path: str = "/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj",
    ) -> None:
        super().__init__()
        self.uv_resolution = uv_resolution
        self.template_mesh_path = template_mesh_path
        self.encoder = DINOv2Encoder(variant="vitb14", freeze=True)
        self.uv_fusion = UVFusionModule()
        self.uv_refinement = UVRefinementNet()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = ["images", "mesh_vertices", "mesh_faces", "intrinsics", "transform_matrices"]
        missing = [k for k in required if k not in batch]
        if missing:
            raise KeyError(f"Stage1PriorBuilder missing required keys: {missing}")

        image_features = self.encoder(batch["images"])

        projection_out = project_image_features_to_surface(
            image_features=image_features,
            mesh_vertices=batch["mesh_vertices"],
            mesh_faces=batch["mesh_faces"],
            intrinsics=batch["intrinsics"],
            transform_matrices=batch["transform_matrices"],
            template_mesh_path=batch.get("template_mesh_path", self.template_mesh_path),
            uv_resolution=self.uv_resolution,
            uv_vertices=batch.get("uv_vertices"),
            uv_faces=batch.get("uv_faces"),
        )

        uv_pack = project_surface_to_uv(
            projection_out=projection_out,
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
            uv_valid_mask=uv_pack["uv_valid_mask"],
            uv_position_map=uv_pack["uv_position_map"],
            uv_normal_map=uv_pack["uv_normal_map"],
        )
        uv_feature_single_raw = uv_pack["uv_feature_single"]
        uv_feature_single_masked = uv_pack["uv_feature_single"] * uv_pack["uv_visibility_single"]
        uv_confidence_single = uv_pack["confidence"][:, 0]
        uv_feature_single_weighted = uv_feature_single_masked * uv_confidence_single

        return {
            "image_features": image_features,
            "uv_features": uv_pack["uv_features"],
            "fused_uv_feature_map": refined_uv,
            "fused_confidence_map": fusion_out["fused_confidence"],
            "fusion_weights": fusion_out["fusion_weights"],
            "uv_valid_mask": geom["uv_valid_mask"],
            "uv_position_map": geom["uv_position_map"],
            "uv_normal_map": geom["uv_normal_map"],
            "canonical_uv": refined_uv,
            "uv_feature_single_raw": uv_feature_single_raw,
            "uv_feature_single_masked": uv_feature_single_masked,
            "uv_feature_single_weighted": uv_feature_single_weighted,
            "uv_confidence_single": uv_confidence_single,
            "uv_feature_single": uv_pack["uv_feature_single"],
            "uv_visibility_single": uv_pack["uv_visibility_single"],
            "uv_face_index": uv_pack["uv_face_index"],
            "uv_barycentric_vis": uv_pack["uv_barycentric_vis"],
            "uv_feature_single_vflip": uv_pack["uv_feature_single_vflip"],
            "uv_sample_xy": uv_pack["uv_sample_xy"],
        }
