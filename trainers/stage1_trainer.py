"""Stage 1 trainer skeleton."""

from __future__ import annotations

from typing import Dict

import torch

from models.stage1_prior.stage1_pipeline import Stage1CanonicalPrior


class Stage1Trainer:
    """Minimal trainer wrapper for Stage 1 experiments."""

    def __init__(self, model: Stage1CanonicalPrior) -> None:
        self.model = model

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one training step.

        TODO: finalize loss terms and weighting.
        """
        outputs = self.model(batch)
        return {"loss": outputs["canonical_uv"].mean() * 0.0}
