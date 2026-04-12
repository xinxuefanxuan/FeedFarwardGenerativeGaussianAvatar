"""Basic mesh operations used by debug scripts."""

from __future__ import annotations

import numpy as np


def mesh_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute simple area-unweighted vertex normals."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return normals / norm


def bbox_2d(points_uv: np.ndarray) -> np.ndarray:
    """Return [xmin, ymin, xmax, ymax] for 2D points."""
    xmin = float(np.min(points_uv[:, 0]))
    ymin = float(np.min(points_uv[:, 1]))
    xmax = float(np.max(points_uv[:, 0]))
    ymax = float(np.max(points_uv[:, 1]))
    return np.asarray([xmin, ymin, xmax, ymax], dtype=np.float32)
