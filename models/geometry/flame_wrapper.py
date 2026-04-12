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


def _to_dense_matrix(m: object) -> np.ndarray:
    """Convert sparse/dense matrix-like object to dense float32 ndarray."""
    if hasattr(m, "toarray"):
        return np.asarray(m.toarray(), dtype=np.float32)
    return np.asarray(m, dtype=np.float32)


def _make_transform(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = rot
    t[:3, 3] = trans
    return t


class FlameWrapper:
    """FLAME wrapper for geometry debugging.

    Phase 2 support in `build_mesh_from_flame_params`:
    - fixed canonical y recenter
    - identity shape deformation (`shape`)
    - expression deformation (`expr`)
    - local joints: neck/jaw/eyes (basic LBS)
    - global rotation (`rotation`)
    - global translation (`translation`)

    Not supported in this phase:
    - posedirs / pose-corrective terms
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

        _, template_faces_obj = _load_obj_vertices_faces(self.head_template_mesh_path)

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

        if "shapedirs" not in flame_data:
            raise KeyError("FLAME pickle missing required key `shapedirs`")
        shapedirs = _to_v3n(np.asarray(flame_data["shapedirs"]), num_vertices=self.num_vertices)

        self.shape_dim = 300
        self.expr_dim = 100
        total_dim = int(shapedirs.shape[2])
        if total_dim < self.shape_dim + self.expr_dim:
            raise ValueError(
                f"shapedirs basis dim {total_dim} is smaller than required {self.shape_dim + self.expr_dim}"
            )

        # Phase 1/2 assumption: first 300 identity, next 100 expression.
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

        # Phase 2: minimal joint chain fields for basic LBS.
        if "J_regressor" not in flame_data or "weights" not in flame_data or "kintree_table" not in flame_data:
            raise KeyError("FLAME pickle must contain `J_regressor`, `weights`, and `kintree_table` for Phase 2")

        self.j_regressor = _to_dense_matrix(flame_data["J_regressor"])
        self.weights = np.asarray(flame_data["weights"], dtype=np.float32)
        self.kintree_table = np.asarray(flame_data["kintree_table"], dtype=np.int64)

        if self.j_regressor.shape[1] != self.num_vertices:
            raise ValueError(
                f"J_regressor shape mismatch: {self.j_regressor.shape}, expected (*,{self.num_vertices})"
            )
        self.num_joints = int(self.j_regressor.shape[0])

        if self.weights.shape[0] != self.num_vertices or self.weights.shape[1] < self.num_joints:
            raise ValueError(
                f"weights shape mismatch: {self.weights.shape}, expected ({self.num_vertices}, >= {self.num_joints})"
            )
        self.weights = self.weights[:, : self.num_joints]

        if self.kintree_table.ndim == 2 and self.kintree_table.shape[0] == 2:
            self.parents = self.kintree_table[0].copy()
        elif self.kintree_table.ndim == 1 and self.kintree_table.shape[0] == self.num_joints:
            self.parents = self.kintree_table.copy()
        else:
            raise ValueError(f"Unsupported kintree_table shape: {self.kintree_table.shape}")

        self.parents = self.parents.astype(np.int64)
        self.parents[0] = -1

        # Explicit, debuggable joint mapping assumption for FLAME-like topology.
        # Supported local joints in Phase 2: neck, jaw, left_eye, right_eye.
        self.joint_ids = {
            "neck": 1,
            "jaw": 2,
            "left_eye": 3,
            "right_eye": 4,
        }
        print(f"[FlameWrapper] Phase-2 joint mapping assumption: {self.joint_ids}")

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

    def _prepare_local_rotations(self, flame_params: Dict[str, np.ndarray]) -> np.ndarray:
        """Build per-joint local rotation matrices for basic LBS.

        Supported joints in this phase:
        - neck: `neck_pose` (1,3)
        - jaw: `jaw_pose` (1,3)
        - eyes: `eyes_pose` (1,6) -> left/right 3D axis-angle

        Unused in this phase:
        - posedirs / pose-corrective terms
        """
        local_rots = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (self.num_joints, 1, 1))

        if "neck_pose" in flame_params:
            neck_pose = np.asarray(flame_params["neck_pose"], dtype=np.float32).reshape(-1)
            if neck_pose.size >= 3 and self.joint_ids["neck"] < self.num_joints:
                local_rots[self.joint_ids["neck"]] = _axis_angle_to_rotation_matrix(neck_pose[:3])

        if "jaw_pose" in flame_params:
            jaw_pose = np.asarray(flame_params["jaw_pose"], dtype=np.float32).reshape(-1)
            if jaw_pose.size >= 3 and self.joint_ids["jaw"] < self.num_joints:
                local_rots[self.joint_ids["jaw"]] = _axis_angle_to_rotation_matrix(jaw_pose[:3])

        if "eyes_pose" in flame_params:
            eyes_pose = np.asarray(flame_params["eyes_pose"], dtype=np.float32).reshape(-1)
            if eyes_pose.size >= 6:
                if self.joint_ids["left_eye"] < self.num_joints:
                    local_rots[self.joint_ids["left_eye"]] = _axis_angle_to_rotation_matrix(eyes_pose[:3])
                if self.joint_ids["right_eye"] < self.num_joints:
                    local_rots[self.joint_ids["right_eye"]] = _axis_angle_to_rotation_matrix(eyes_pose[3:6])

        return local_rots

    def _compute_joints(self, vertices: np.ndarray) -> np.ndarray:
        """Compute joints from vertices using J_regressor."""
        # J_regressor: (J, V), vertices: (V, 3) -> joints: (J, 3)
        return self.j_regressor @ vertices

    def _basic_lbs(self, vertices: np.ndarray, local_rots: np.ndarray) -> np.ndarray:
        """Apply basic LBS without posedirs / pose-corrective."""
        joints = self._compute_joints(vertices)

        transforms = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (self.num_joints, 1, 1))

        for j in range(self.num_joints):
            parent = int(self.parents[j])
            if parent < 0:
                transforms[j] = _make_transform(local_rots[j], joints[j])
            else:
                rel_t = joints[j] - joints[parent]
                transforms[j] = transforms[parent] @ _make_transform(local_rots[j], rel_t)

        # Remove rest joint offsets: G_j = T_j * [I, -J_j]
        rest = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (self.num_joints, 1, 1))
        rest[:, :3, 3] = -joints
        transforms = np.matmul(transforms, rest)

        # Blend transforms per vertex.
        blended = np.einsum("vj,jab->vab", self.weights, transforms)
        verts_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
        posed_h = np.einsum("vab,vb->va", blended, verts_h)
        return posed_h[:, :3]

    def build_mesh_from_flame_params(self, flame_params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return Phase-2 FLAME mesh for visualization.

        Supported in this phase:
        1) fixed canonical y recenter
        2) identity shape deformation (300)
        3) expression deformation (100)
        4) local joints neck/jaw/eyes with basic LBS
        5) global rotation
        6) global translation

        Explicitly unsupported in this phase:
        - posedirs / pose-corrective terms
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

        # 4) local joints + basic LBS (no posedirs)
        local_rots = self._prepare_local_rotations(flame_params)
        vertices = self._basic_lbs(vertices, local_rots)

        # 5) global rotation (axis-angle / Rodrigues)
        rotation = flame_params.get("rotation")
        if rotation is not None:
            rot_mat = _axis_angle_to_rotation_matrix(rotation)
            vertices = (rot_mat @ vertices.T).T

        # 6) global translation
        translation = flame_params.get("translation")
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float32).reshape(-1)
            if translation.size >= 3:
                vertices = vertices + translation[:3][None, :]

        return vertices.astype(np.float32), self.template_faces
