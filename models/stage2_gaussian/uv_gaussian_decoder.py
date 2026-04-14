"""Decode UV canonical priors into a minimal Gaussian field."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.stage2_gaussian.gaussian_head import GaussianParamHead


class UVGaussianDecoder(nn.Module):
    """Geometry-first Gaussian decoder from canonical UV maps."""

    def __init__(self, uv_feature_dim: int = 32, hidden_dim: int = 128, color_dim: int = 16) -> None:
        super().__init__()
        in_dim = 3 + 3 + uv_feature_dim + 1 + 1  # pos + normal + uv feat + conf + valid
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = GaussianParamHead(latent_dim=hidden_dim, color_dim=color_dim)

    def forward(
        self,
        uv_valid_mask: torch.Tensor,
        uv_position_map: torch.Tensor,
        uv_normal_map: torch.Tensor,
        uv_feature_map: torch.Tensor,
        uv_confidence_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, _, h, w = uv_valid_mask.shape
        device = uv_valid_mask.device

        valid0 = (uv_valid_mask[0, 0] > 0.5).view(-1)
        uv_index = torch.where(valid0)[0]
        if uv_index.numel() == 0:
            raise RuntimeError("UVGaussianDecoder found zero valid UV pixels.")
        g = int(uv_index.numel())

        pos_flat = uv_position_map.view(b, 3, -1).transpose(1, 2)
        nrm_flat = uv_normal_map.view(b, 3, -1).transpose(1, 2)
        feat_flat = uv_feature_map.view(b, uv_feature_map.shape[1], -1).transpose(1, 2)
        conf_flat = uv_confidence_map.view(b, 1, -1).transpose(1, 2)
        valid_flat = uv_valid_mask.view(b, 1, -1).transpose(1, 2)

        pos = pos_flat[:, uv_index]
        nrm = nrm_flat[:, uv_index]
        feat = feat_flat[:, uv_index]
        conf = conf_flat[:, uv_index]
        valid = valid_flat[:, uv_index]

        x = torch.cat([pos, nrm, feat, conf, valid], dim=-1)
        latent = self.backbone(x)
        raw = self.head(latent)

        delta_xyz = 0.01 * torch.tanh(raw["delta_xyz"])
        xyz = pos + delta_xyz
        scale = F.softplus(raw["scale_raw"]) + 1.0e-3
        rotation = F.normalize(raw["rotation_raw"], dim=-1, eps=1.0e-8)
        opacity = torch.sigmoid(raw["opacity_raw"])
        color_feat = torch.tanh(raw["color_feat_raw"])

        return {
            "gaussian_xyz": xyz,                            # [B,G,3]
            "gaussian_scale": scale,                        # [B,G,3]
            "gaussian_rotation": rotation,                  # [B,G,4]
            "gaussian_opacity": opacity,                    # [B,G,1]
            "gaussian_color_feat": color_feat,              # [B,G,Cg]
            "gaussian_uv_index": uv_index.unsqueeze(0).repeat(b, 1).to(device=device),  # [B,G]
        }
