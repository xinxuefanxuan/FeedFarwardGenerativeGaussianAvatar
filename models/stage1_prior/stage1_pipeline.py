"""Stage 1 pipeline entry wrapping the MVP prior builder."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.stage1_prior.prior_builder import Stage1PriorBuilder


class Stage1CanonicalPrior(nn.Module):
    """Build geometry-consistent canonical priors from sparse observations."""

    def __init__(self, uv_resolution: int = 256) -> None:
        super().__init__()
        self.builder = Stage1PriorBuilder(uv_resolution=uv_resolution)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.builder(batch)
