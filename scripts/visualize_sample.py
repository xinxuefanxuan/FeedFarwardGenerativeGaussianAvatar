"""Visualize one real processed FastAvatar-style sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.nersemble_dataset import NersembleFastAvatarDataset


def _to_numpy(x: object) -> np.ndarray:
    """Convert numpy/torch-like input to numpy array without changing values."""
    if isinstance(x, np.ndarray):
        return x

    # Torch-like tensor support without hard dependency on torch import.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _normalize_rgb_layout(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB layout to HWC/HW for PIL saving."""
    arr = np.squeeze(rgb)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # single channel HWC -> HW
        if arr.shape[-1] == 1:
            arr = arr[..., 0]

    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported rgb shape after normalization: {arr.shape}")

    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected last channel in (3,4), got shape {arr.shape}")

    return arr


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert image-like array to uint8."""
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.max(arr)) if arr.size > 0 else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_rgb(rgb: object, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _to_numpy(rgb)
    arr = _normalize_rgb_layout(arr)
    arr = _to_uint8(arr)

    print(
        f"[visualize_sample] saving rgb shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min() if arr.size else 'NA'}, max={arr.max() if arr.size else 'NA'}"
    )

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save visualization images") from exc

    Image.fromarray(arr).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-dir",
        type=str,
        default="/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full/017/EXP-1-head_part-1/cam_220700191",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/visualize/sample_rgb.png")
    args = parser.parse_args()

    dataset = NersembleFastAvatarDataset(camera_dir=args.camera_dir)
    sample = dataset[args.index]

    _save_rgb(sample["rgb"], Path(args.out))

    print("Saved RGB visualization:", args.out)
    print("Intrinsics:\n", sample["intrinsics"])
    print("Transform matrix:\n", sample["transform_matrix"])
    print("Flame fields:", sorted(sample["flame_params"].keys()))


if __name__ == "__main__":
    main()
