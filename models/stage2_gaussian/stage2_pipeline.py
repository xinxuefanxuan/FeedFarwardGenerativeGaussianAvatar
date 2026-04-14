"""Stage 2 feedforward Gaussian avatar MVP pipeline."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.stage2_gaussian.render_bridge import Stage2RenderBridge
from models.stage2_gaussian.stage2_input_adapter import Stage2InputBatch, stage2_batch_from_dict
from models.stage2_gaussian.uv_gaussian_decoder import UVGaussianDecoder


class Stage2GaussianAvatar(nn.Module):
    """Generate and render Gaussian avatar from canonical prior maps."""

    def __init__(self, uv_feature_dim: int = 32, color_dim: int = 16) -> None:
        super().__init__()
        self.decoder = UVGaussianDecoder(uv_feature_dim=uv_feature_dim, color_dim=color_dim)
        self.render_bridge = Stage2RenderBridge()

    def forward(self, batch_or_dict: Dict[str, torch.Tensor] | Stage2InputBatch) -> Dict[str, torch.Tensor]:
        batch = stage2_batch_from_dict(batch_or_dict) if isinstance(batch_or_dict, dict) else batch_or_dict

        gaussians = self.decoder(
            uv_valid_mask=batch.uv_valid_mask,
            uv_position_map=batch.uv_position_map,
            uv_normal_map=batch.uv_normal_map,
            uv_feature_map=batch.uv_feature_map,
            uv_confidence_map=batch.uv_confidence_map,
        )
        _, _, _, h, w = batch.target_images.shape
        render_out = self.render_bridge(
            gaussians=gaussians,
            intrinsics=batch.intrinsics,
            extrinsics=batch.extrinsics,
            image_size=(h, w),
        )
        return {**gaussians, **render_out}
