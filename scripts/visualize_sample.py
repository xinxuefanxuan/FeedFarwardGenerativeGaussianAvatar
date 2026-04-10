"""Visualize one real processed FastAvatar-style sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.nersemble_dataset import NersembleFastAvatarDataset


def _save_rgb(rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save visualization images") from exc

    Image.fromarray(rgb_u8).save(out_path)


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
