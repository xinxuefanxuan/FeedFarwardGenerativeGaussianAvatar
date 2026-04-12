"""FLAME asset loading and coarse mesh generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _load_obj_vertices_faces(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle Rodrigues vector to a 3x3 rotation matrix."""
    vec = np.asarray(axis_angle, dtype=np.float32).reshape(-1)
    if vec.size < 3:
        return np.eye(3, dtype=np.float32)

    r = vec[:3]
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)

    k = r / theta
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    k_mat = np.asarray(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float32,
    )
    ident = np.eye(3, dtype=np.float32)
    return ident + np.sin(theta) * k_mat + (1.0 - np.cos(theta)) * (k_mat @ k_mat)


class FlameWrapper:
    """Thin wrapper around FLAME assets for geometry debugging.

    Note:
    - This round focuses on stable data + geometry infrastructure.
    - Full FLAME deformation is intentionally deferred.
    """

    def __init__(
        self,
        flame_model_path: str | Path,
        head_template_mesh_path: str | Path,
        masks_path: str | Path,
        uv_masks_path: str | Path,
        recenter_head_y: bool = True,
    ) -> None:
        self.flame_model_path = Path(flame_model_path)
        self.head_template_mesh_path = Path(head_template_mesh_path)
        self.masks_path = Path(masks_path)
        self.uv_masks_path = Path(uv_masks_path)
        self.recenter_head_y = recenter_head_y

        self.template_vertices, self.template_faces = _load_obj_vertices_faces(self.head_template_mesh_path)
        # Fixed template center; do NOT use per-frame dynamic mean.
        self.template_head_y_mean = float(self.template_vertices[:, 1].mean())

    def validate_assets(self) -> None:
        """Raise if required FLAME assets are missing."""
        required = [
            self.flame_model_path,
            self.head_template_mesh_path,
            self.masks_path,
            self.uv_masks_path,
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing FLAME assets: {missing}")

    def build_mesh_from_flame_params(self, flame_params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return coarse mesh for visualization from FLAME params.

        Current behavior:
        - Uses head template mesh as base geometry.
        - Applies fixed template y recentering.
        - Applies global rotation (axis-angle / Rodrigues).
        - Applies global translation.

        TODO: replace with full FLAME forward when model/runtime is integrated.
        """
        vertices = self.template_vertices.copy()

        # 1) fixed template recenter (not per-frame dynamic mean)
        if self.recenter_head_y:
            vertices[:, 1] -= self.template_head_y_mean

        # 2) global rotation
        rotation = flame_params.get("rotation")
        if rotation is not None:
            rot_mat = _axis_angle_to_rotation_matrix(rotation)
            vertices = (rot_mat @ vertices.T).T

        # 3) global translation
        translation = flame_params.get("translation")
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float32).reshape(-1)
            if translation.size >= 3:
                vertices = vertices + translation[:3][None, :]

        return vertices, self.template_faces
