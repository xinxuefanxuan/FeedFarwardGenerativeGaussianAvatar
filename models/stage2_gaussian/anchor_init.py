"""Gaussian anchor initialization placeholders."""

from __future__ import annotations

import torch


def initialize_gaussian_anchors(source_tensor: torch.Tensor, num_anchors: int) -> torch.Tensor:
    """Initialize Gaussian anchors from UV or mesh signals.

    TODO: confirm anchor sampling policy.
    """
    if source_tensor.numel() == 0:
        raise ValueError("source_tensor should not be empty")
    return source_tensor.reshape(-1, source_tensor.shape[-1])[:num_anchors]
