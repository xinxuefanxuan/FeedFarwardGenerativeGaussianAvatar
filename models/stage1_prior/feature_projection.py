"""Project image features onto coarse FLAME mesh."""

from __future__ import annotations

import torch


def project_features_to_mesh(
    image_features: torch.Tensor,
    coarse_mesh: torch.Tensor,
    camera_params: torch.Tensor,
) -> torch.Tensor:
    """Project image features onto mesh vertices.

    TODO: finalize projection method after confirming camera convention.
    """
    raise NotImplementedError
