"""Mesh-to-image projection interfaces."""

from __future__ import annotations

import numpy as np

from models.geometry.camera_utils import project_points


def project_mesh_vertices(
    vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    transform_matrix: np.ndarray,
    transform_mode: str = "unknown",
) -> np.ndarray:
    """Project mesh vertices to 2D image coordinates."""
    return project_points(vertices_world, intrinsics, transform_matrix, transform_mode=transform_mode)
