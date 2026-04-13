"""Stage 2 trainer skeleton."""

from __future__ import annotations

from typing import Dict

import torch

from models.stage2_gaussian.stage2_pipeline import Stage2GaussianAvatar


class Stage2Trainer:
    """Minimal trainer wrapper for Stage 2 experiments."""

    def __init__(self, model: Stage2GaussianAvatar) -> None:
        self.model = model

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one Stage 2 training step.

        TODO: finalize render and regularization losses.
        """
        outputs = self.model(batch)
        return {"loss": outputs["render"].mean() * 0.0}
