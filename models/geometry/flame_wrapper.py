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
    ) -> None:
        self.flame_model_path = Path(flame_model_path)
        self.head_template_mesh_path = Path(head_template_mesh_path)
        self.masks_path = Path(masks_path)
        self.uv_masks_path = Path(uv_masks_path)

        self.template_vertices, self.template_faces = _load_obj_vertices_faces(self.head_template_mesh_path)

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
        - Applies only global translation when provided.

        TODO: replace with full FLAME forward when model/runtime is integrated.
        """
        vertices = self.template_vertices.copy()
        translation = flame_params.get("translation")
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float32).reshape(-1)
            if translation.size >= 3:
                vertices = vertices + translation[:3][None, :]

        return vertices, self.template_faces
