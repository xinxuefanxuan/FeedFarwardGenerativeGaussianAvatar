"""FLAME asset loading and coarse mesh generation utilities."""

from __future__ import annotations

import pickle
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


def _to_v3n(arr: np.ndarray, num_vertices: int) -> np.ndarray:
    """Normalize shape bases to (V, 3, N)."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == num_vertices and a.shape[1] == 3:
        return a.astype(np.float32)
    if a.ndim == 2 and a.shape[0] == num_vertices * 3:
        return a.reshape(num_vertices, 3, -1).astype(np.float32)
    raise ValueError(f"Unsupported basis shape {a.shape}, expected (V,3,N) or (V*3,N)")


def _apply_basis(basis_v3n: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Apply basis (V,3,N) with coeff (N,) -> (V,3)."""
    c = np.asarray(coeff, dtype=np.float32).reshape(-1)
    if basis_v3n.shape[2] != c.shape[0]:
        raise ValueError(f"Basis dim {basis_v3n.shape[2]} != coeff dim {c.shape[0]}")
    return np.tensordot(basis_v3n, c, axes=([2], [0]))


class FlameWrapper:
    """Thin wrapper around FLAME assets for geometry debugging.

    Phase 1 support in `build_mesh_from_flame_params`:
    - fixed canonical y recenter
    - identity shape deformation (`shape`)
    - expression deformation (`expr`)
    - global rotation (`rotation`)
    - global translation (`translation`)

    Not supported in this phase:
    - neck_pose, jaw_pose, eyes_pose
    - full LBS / pose corrective
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

        template_vertices_obj, template_faces_obj = _load_obj_vertices_faces(self.head_template_mesh_path)

        with open(self.flame_model_path, "rb") as f:
            flame_data = pickle.load(f, encoding="latin1")

        if not isinstance(flame_data, dict):
            raise TypeError(f"Expected flame model pickle dict, got {type(flame_data)}")

        if "v_template" not in flame_data:
            raise KeyError("FLAME pickle missing required key `v_template`")
        self.v_template = np.asarray(flame_data["v_template"], dtype=np.float32)
        if self.v_template.ndim != 2 or self.v_template.shape[1] != 3:
            raise ValueError(f"Invalid v_template shape: {self.v_template.shape}")

        self.num_vertices = int(self.v_template.shape[0])
        self.template_vertices = self.v_template.copy()

        if "f" in flame_data:
            self.template_faces = np.asarray(flame_data["f"], dtype=np.int32)
        else:
            self.template_faces = template_faces_obj

        # shapedirs can contain both identity and expression components.
        if "shapedirs" not in flame_data:
            raise KeyError("FLAME pickle missing required key `shapedirs`")
        shapedirs = _to_v3n(np.asarray(flame_data["shapedirs"]), num_vertices=self.num_vertices)

        # Confirmed input protocol currently uses 300 shape + 100 expr coefficients.
        self.shape_dim = 300
        self.expr_dim = 100
        total_dim = int(shapedirs.shape[2])
        if total_dim < self.shape_dim + self.expr_dim:
            raise ValueError(
                f"shapedirs basis dim {total_dim} is smaller than required {self.shape_dim + self.expr_dim}"
            )

        # Assumption (explicit): first 300 = identity shape, next 100 = expression.
        self.shape_basis = shapedirs[:, :, : self.shape_dim]
        self.expr_basis = shapedirs[:, :, self.shape_dim : self.shape_dim + self.expr_dim]

        if "exprdirs" in flame_data:
            exprdirs = _to_v3n(np.asarray(flame_data["exprdirs"]), num_vertices=self.num_vertices)
            if exprdirs.shape[2] >= self.expr_dim:
                self.expr_basis = exprdirs[:, :, : self.expr_dim]
                print(
                    "[FlameWrapper] Using `exprdirs` for expression basis (first 100 dims); "
                    "shape basis remains first 300 dims from `shapedirs`."
                )
            else:
                print(
                    f"[FlameWrapper] `exprdirs` exists but dim={exprdirs.shape[2]} < 100; "
                    "fall back to shapedirs[300:400] for expression."
                )
        else:
            print(
                "[FlameWrapper] Assumption: expression basis uses shapedirs[:, :, 300:400]. "
                "(No `exprdirs` key found in FLAME pickle.)"
            )

        # Fixed canonical center; do NOT use per-frame dynamic mean.
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

    def _prepare_coeff(self, flame_params: Dict[str, np.ndarray], key: str, expected_dim: int) -> np.ndarray:
        if key not in flame_params:
            raise KeyError(f"Missing required FLAME param `{key}`")
        coeff = np.asarray(flame_params[key], dtype=np.float32).reshape(-1)
        if coeff.shape[0] != expected_dim:
            raise ValueError(f"`{key}` dim mismatch: got {coeff.shape[0]}, expected {expected_dim}")
        return coeff

    def build_mesh_from_flame_params(self, flame_params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return Phase-1 FLAME mesh for visualization.

        Supported in this phase:
        1) fixed canonical y recenter
        2) identity shape deformation (300)
        3) expression deformation (100)
        4) global rotation
        5) global translation

        Explicitly unsupported in this phase:
        - neck_pose / jaw_pose / eyes_pose
        - full LBS and pose corrective terms
        """
        vertices = self.v_template.copy()

        # 1) fixed canonical recenter (not per-frame dynamic mean)
        if self.recenter_head_y:
            vertices[:, 1] -= self.template_head_y_mean

        # 2) identity shape deformation
        shape_coeff = self._prepare_coeff(flame_params, "shape", self.shape_dim)
        vertices = vertices + _apply_basis(self.shape_basis, shape_coeff)

        # 3) expression deformation
        expr_coeff = self._prepare_coeff(flame_params, "expr", self.expr_dim)
        vertices = vertices + _apply_basis(self.expr_basis, expr_coeff)

        # 4) global rotation (axis-angle / Rodrigues)
        rotation = flame_params.get("rotation")
        if rotation is not None:
            rot_mat = _axis_angle_to_rotation_matrix(rotation)
            vertices = (rot_mat @ vertices.T).T

        # 5) global translation
        translation = flame_params.get("translation")
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float32).reshape(-1)
            if translation.size >= 3:
                vertices = vertices + translation[:3][None, :]

        return vertices.astype(np.float32), self.template_faces
