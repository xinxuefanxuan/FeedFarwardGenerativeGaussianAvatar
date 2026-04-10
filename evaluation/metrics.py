"""Evaluation metric placeholders."""

from __future__ import annotations

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Placeholder PSNR metric."""
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse + 1e-8)
