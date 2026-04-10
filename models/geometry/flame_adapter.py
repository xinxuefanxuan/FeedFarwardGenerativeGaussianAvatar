"""FLAME adapter and projection placeholders."""

from __future__ import annotations

from typing import Dict

import torch


class FlameAdapter:
    """Utility wrapper for FLAME parameters and mesh conversion.

    TODO: confirm FLAME parameter names and coordinate conventions.
    """

    def build_coarse_mesh(self, flame_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build a coarse mesh from FLAME tracking parameters."""
        raise NotImplementedError
