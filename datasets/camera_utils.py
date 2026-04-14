"""Camera utilities for transforms.json schema."""

from __future__ import annotations

from typing import Dict

import numpy as np


def intrinsics_from_frame(frame: Dict[str, float]) -> np.ndarray:
    """Build a 3x3 intrinsic matrix from frame/top-level camera fields."""
    fl_x = float(frame["fl_x"])
    fl_y = float(frame["fl_y"])
    cx = float(frame["cx"])
    cy = float(frame["cy"])

    k = np.eye(3, dtype=np.float32)
    k[0, 0] = fl_x
    k[1, 1] = fl_y
    k[0, 2] = cx
    k[1, 2] = cy
    return k


def transform_matrix_from_frame(frame: Dict[str, object]) -> np.ndarray:
    """Load 4x4 transform matrix as float32 array."""
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected transform_matrix shape (4, 4), got {matrix.shape}")
    return matrix


def guess_transform_mode(transform: np.ndarray) -> str:
    """Heuristic guess for transform convention.

    Returns:
        'likely_cam2world', 'likely_world2cam', or 'unknown'.
    """
    if transform.shape != (4, 4):
        return "unknown"

    rotation = transform[:3, :3]
    det_r = np.linalg.det(rotation)
    if not np.isfinite(det_r):
        return "unknown"

    if np.abs(det_r - 1.0) < 1e-2:
        # Both cam2world and world2cam can satisfy this; keep explicit.
        return "unknown"
    return "unknown"
