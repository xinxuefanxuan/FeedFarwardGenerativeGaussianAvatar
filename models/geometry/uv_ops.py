"""UV utility placeholders for geometry maps."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_uv_valid_mask(uv_resolution: int = 256) -> np.ndarray:
    """Create a placeholder UV valid mask.

    TODO: replace with FLAME UV mask from `uv_masks.npz` semantics.
    """
    return np.ones((uv_resolution, uv_resolution), dtype=np.float32)


def build_geometry_maps_placeholder(
    uv_resolution: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return placeholder position and normal maps in UV space."""
    pos = np.zeros((uv_resolution, uv_resolution, 3), dtype=np.float32)
    nrm = np.zeros((uv_resolution, uv_resolution, 3), dtype=np.float32)
    nrm[..., 2] = 1.0
    return pos, nrm
