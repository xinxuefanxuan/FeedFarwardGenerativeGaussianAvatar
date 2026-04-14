"""Minimal Stage-2 losses for debug training."""

from __future__ import annotations

from typing import Dict

import torch


def stage2_mvp_losses(
    rendered_images: torch.Tensor,
    rendered_alpha: torch.Tensor,
    target_images: torch.Tensor,
    target_masks: torch.Tensor | None,
    gaussian_xyz: torch.Tensor,
    gaussian_scale: torch.Tensor,
    gaussian_opacity: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    loss_image = torch.mean(torch.abs(rendered_images - target_images))
    if target_masks is None:
        loss_alpha = rendered_alpha.mean() * 0.0
    else:
        loss_alpha = torch.mean(torch.abs(rendered_alpha - target_masks))

    loss_xyz_reg = torch.mean(torch.norm(gaussian_xyz, dim=-1))
    loss_scale_reg = torch.mean(gaussian_scale)
    loss_opacity_reg = torch.mean(gaussian_opacity)

    total = (
        loss_image
        + 0.1 * loss_alpha
        + 1.0e-3 * loss_xyz_reg
        + 1.0e-3 * loss_scale_reg
        + 1.0e-3 * loss_opacity_reg
    )
    return {
        "loss_total": total,
        "loss_image_l1": loss_image,
        "loss_alpha": loss_alpha,
        "loss_xyz_reg": loss_xyz_reg,
        "loss_scale_reg": loss_scale_reg,
        "loss_opacity_reg": loss_opacity_reg,
    }
