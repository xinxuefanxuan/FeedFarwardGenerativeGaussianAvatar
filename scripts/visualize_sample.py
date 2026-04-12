"""Visualize one real processed FastAvatar-style sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.nersemble_dataset import NersembleFastAvatarDataset


# def _save_rgb(rgb: np.ndarray, out_path: Path) -> None:
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
#     try:
#         from PIL import Image
#     except ImportError as exc:
#         raise ImportError("Pillow is required to save visualization images") from exc

#     Image.fromarray(rgb_u8).save(out_path)
def _save_rgb(rgb, out_path):
    import numpy as np
    from pathlib import Path
    from PIL import Image

    try:
        import torch
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
    except Exception:
        pass

    rgb = np.asarray(rgb)
    print("[DEBUG] before save:", rgb.shape, rgb.dtype, rgb.min(), rgb.max())

    # 去掉 batch 维
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]

    # CHW -> HWC
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (1, 3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))

    # 单通道 squeeze
    if rgb.ndim == 3 and rgb.shape[-1] == 1:
        rgb = rgb[..., 0]

    # float -> uint8
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb.clip(0, 255).astype(np.uint8)

    print("[DEBUG] after prep:", rgb.shape, rgb.dtype, rgb.min(), rgb.max())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)

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
