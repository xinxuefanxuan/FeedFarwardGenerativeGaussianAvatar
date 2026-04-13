"""Shared camera math helpers for geometry pipeline."""

from __future__ import annotations

import numpy as np


def to_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    """Convert Nx3 points to Nx4 homogeneous coordinates."""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)


def project_points(
    vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    transform_matrix: np.ndarray,
    transform_mode: str = "unknown",
) -> np.ndarray:
    """Project 3D points to image plane.

    Args:
        transform_mode: 'cam2world', 'world2cam', or 'unknown'.
            If unknown, defaults to interpreting matrix as world2cam for now.
    """
    vertices_h = to_homogeneous(vertices_world)

    if transform_mode == "cam2world":
        world2cam = np.linalg.inv(transform_matrix)
    else:
        world2cam = transform_matrix

    cam = (world2cam @ vertices_h.T).T[:, :3]
    z = np.clip(cam[:, 2:3], 1e-6, None)
    x = cam[:, 0:1] / z
    y = cam[:, 1:2] / z

    uv_h = (intrinsics @ np.concatenate([x, y, np.ones_like(x)], axis=1).T).T
    return uv_h[:, :2]
