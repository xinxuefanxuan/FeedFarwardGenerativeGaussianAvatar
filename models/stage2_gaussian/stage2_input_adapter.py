"""Stage-2 input schema and adapter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class Stage2InputBatch:
    """Canonical-prior conditioned Stage-2 batch."""

    uv_valid_mask: torch.Tensor          # [B,1,Huv,Wuv]
    uv_position_map: torch.Tensor        # [B,3,Huv,Wuv]
    uv_normal_map: torch.Tensor          # [B,3,Huv,Wuv]
    uv_feature_map: torch.Tensor         # [B,Cuv,Huv,Wuv]
    uv_confidence_map: torch.Tensor      # [B,1,Huv,Wuv]
    target_images: torch.Tensor          # [B,V,3,Himg,Wimg]
    intrinsics: torch.Tensor             # [B,V,3,3]
    extrinsics: torch.Tensor             # [B,V,4,4]
    target_masks: Optional[torch.Tensor] = None  # [B,V,1,Himg,Wimg]
    mesh_faces: Optional[torch.Tensor] = None


def stage2_batch_from_dict(batch: Dict[str, torch.Tensor]) -> Stage2InputBatch:
    required = [
        "uv_valid_mask",
        "uv_position_map",
        "uv_normal_map",
        "uv_feature_map",
        "uv_confidence_map",
        "target_images",
        "intrinsics",
        "extrinsics",
    ]
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(f"Stage2InputBatch missing required keys: {missing}")
    return Stage2InputBatch(
        uv_valid_mask=batch["uv_valid_mask"],
        uv_position_map=batch["uv_position_map"],
        uv_normal_map=batch["uv_normal_map"],
        uv_feature_map=batch["uv_feature_map"],
        uv_confidence_map=batch["uv_confidence_map"],
        target_images=batch["target_images"],
        intrinsics=batch["intrinsics"],
        extrinsics=batch["extrinsics"],
        target_masks=batch.get("target_masks"),
        mesh_faces=batch.get("mesh_faces"),
    )
