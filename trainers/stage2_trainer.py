"""Stage 2 trainer skeleton."""

from __future__ import annotations

from typing import Dict

import torch

from models.stage2_gaussian.losses import stage2_mvp_losses
from models.stage2_gaussian.stage2_pipeline import Stage2GaussianAvatar


class Stage2Trainer:
    """Minimal trainer wrapper for Stage 2 experiments."""

    def __init__(self, model: Stage2GaussianAvatar) -> None:
        self.model = model

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one Stage 2 MVP training step."""
        outputs = self.model(batch)
        losses = stage2_mvp_losses(
            rendered_images=outputs["rendered_images"],
            rendered_alpha=outputs["rendered_alpha"],
            target_images=batch["target_images"],
            target_masks=batch.get("target_masks"),
            gaussian_xyz=outputs["gaussian_xyz"],
            gaussian_scale=outputs["gaussian_scale"],
            gaussian_opacity=outputs["gaussian_opacity"],
        )
        return {"loss": losses["loss_total"], "losses": losses, "outputs": outputs}
