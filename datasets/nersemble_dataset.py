"""Primary dataset implementation for processed FastAvatar-style NeRSemble data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from datasets.base_dataset import BaseAvatarDataset
from datasets.camera_utils import intrinsics_from_frame, transform_matrix_from_frame


class NersembleFastAvatarDataset(BaseAvatarDataset):
    """Load one camera sequence using confirmed transforms.json schema.

    Expected root structure (confirmed sample):
    `<root>/<subject>/<sequence>/<camera>/transforms.json`
    """

    def __init__(
        self,
        camera_dir: str | Path,
        split: str = "train",
        prefer_rgb_npy: bool = True,
        processed_data_dirname: str = "processed_data",
    ) -> None:
        super().__init__(split=split)
        self.camera_dir = Path(camera_dir)
        self.transforms_path = self.camera_dir / "transforms.json"
        self.prefer_rgb_npy = prefer_rgb_npy
        self.processed_data_dirname = processed_data_dirname

        if not self.transforms_path.exists():
            raise FileNotFoundError(f"Missing transforms.json: {self.transforms_path}")

        with open(self.transforms_path, "r", encoding="utf-8") as f:
            self.transforms = json.load(f)

        self.frames: List[Dict[str, Any]] = self.transforms["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def _frame_processed_dir(self, frame: Dict[str, Any]) -> Path:
        timestep_id = str(frame["timestep_id"])
        return self.camera_dir / self.processed_data_dirname / timestep_id

    def _load_rgb(self, frame: Dict[str, Any]) -> np.ndarray:
        processed_dir = self._frame_processed_dir(frame)
        rgb_npy = processed_dir / "rgb.npy"
        if self.prefer_rgb_npy and rgb_npy.exists():
            rgb = np.load(rgb_npy)
            return rgb.astype(np.float32)

        img_rel = frame.get("file_path")
        if img_rel is None:
            raise KeyError("Frame missing file_path, cannot load RGB fallback image")

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for image fallback loading") from exc

        img_path = self.camera_dir / str(img_rel)
        with Image.open(img_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32)

    def _load_mask(self, frame: Dict[str, Any]) -> Optional[np.ndarray]:
        processed_dir = self._frame_processed_dir(frame)
        mask_npy = processed_dir / "mask.npy"
        if mask_npy.exists():
            return np.load(mask_npy).astype(np.float32)

        mask_rel = frame.get("fg_mask_path")
        if mask_rel is None:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        mask_path = self.camera_dir / str(mask_rel)
        with Image.open(mask_path) as m:
            return np.asarray(m, dtype=np.float32)

    def _load_intrinsics(self, frame: Dict[str, Any]) -> np.ndarray:
        processed_dir = self._frame_processed_dir(frame)
        intrs_npy = processed_dir / "intrs.npy"
        if intrs_npy.exists():
            intrs = np.load(intrs_npy).astype(np.float32)
            if intrs.shape == (3, 3):
                return intrs
        return intrinsics_from_frame(frame)

    def _load_landmark2d(self, frame: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        processed_dir = self._frame_processed_dir(frame)
        lm_path = processed_dir / "landmark2d.npz"
        if not lm_path.exists():
            return None
        data = np.load(lm_path)
        return {k: data[k] for k in data.files}

    def _load_flame_params(self, frame: Dict[str, Any]) -> Dict[str, np.ndarray]:
        flame_rel = frame.get("flame_param_path")
        if flame_rel is None:
            raise KeyError("Frame missing flame_param_path")

        flame_path = self.camera_dir / str(flame_rel)
        if not flame_path.exists():
            raise FileNotFoundError(f"Missing flame params file: {flame_path}")

        flame_data = np.load(flame_path)
        return {k: flame_data[k] for k in flame_data.files}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame = self.frames[index]

        return {
            "rgb": self._load_rgb(frame),
            "mask": self._load_mask(frame),
            "intrinsics": self._load_intrinsics(frame),
            "transform_matrix": transform_matrix_from_frame(frame),
            "landmark2d": self._load_landmark2d(frame),
            "flame_params": self._load_flame_params(frame),
            "frame_meta": frame,
            "scene_meta": {
                "camera_indices": self.transforms.get("camera_indices"),
                "timestep_indices": self.transforms.get("timestep_indices"),
            },
        }


class NersembleRawAdapter(BaseAvatarDataset):
    """Raw NeRSemble adapter placeholder.

    Current confirmed fact:
    - camera_params are organized by subject, e.g. `<raw_root>/camera_params/018`

    TODO: finalize raw schema parsing once additional fields are confirmed.
    """

    def __init__(self, raw_root: str | Path, subject_id: str, split: str = "train") -> None:
        super().__init__(split=split)
        self.raw_root = Path(raw_root)
        self.subject_id = subject_id

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError("Raw NeRSemble adapter not implemented: schema still pending.")
