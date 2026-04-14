"""Visibility helpers for projected mesh points."""

from __future__ import annotations

import numpy as np


def points_in_image_mask(points_uv: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Return bool mask for points inside image bounds."""
    h, w = image_hw
    x = points_uv[:, 0]
    y = points_uv[:, 1]
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)
