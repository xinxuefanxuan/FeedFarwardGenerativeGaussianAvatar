"""Debug FLAME mesh overlay on a real sample image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.nersemble_dataset import NersembleFastAvatarDataset
from models.geometry.flame_wrapper import FlameWrapper
from models.geometry.mesh_ops import mesh_vertex_normals
from models.geometry.projection import project_mesh_vertices
from models.geometry.uv_ops import build_geometry_maps_placeholder, build_uv_valid_mask
from models.geometry.visibility import points_in_image_mask


def _draw_points(rgb: np.ndarray, points_uv: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.clip(rgb, 0, 255).astype(np.uint8).copy()
    for (u, v), is_valid in zip(points_uv, valid_mask):
        if not is_valid:
            continue
        x = int(round(u))
        y = int(round(v))
        if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
            canvas[y, x] = np.array([255, 0, 0], dtype=np.uint8)
    return canvas


def _save_image(img, path):
    import numpy as np
    from pathlib import Path
    from PIL import Image

    try:
        import torch
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
    except Exception:
        pass

    img = np.asarray(img)
    print("[DEBUG overlay] before save:", img.shape, img.dtype, img.min(), img.max())

    # 去掉 batch 维
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    # 单通道 squeeze
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    # float -> uint8
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    print("[DEBUG overlay] after prep:", img.shape, img.dtype, img.min(), img.max())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-dir",
        type=str,
        default="/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full/017/EXP-1-head_part-1/cam_220700191",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--flame-model", type=str, default="/home/yuanyuhao/VHAP/asset/flame/flame2023.pkl")
    parser.add_argument("--flame-template", type=str, default="/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj")
    parser.add_argument("--flame-masks", type=str, default="/home/yuanyuhao/VHAP/asset/flame/FLAME_masks.pkl")
    parser.add_argument("--uv-masks", type=str, default="/home/yuanyuhao/VHAP/asset/flame/uv_masks.npz")
    parser.add_argument("--uv-resolution", type=int, default=256)
    parser.add_argument("--out-overlay", type=str, default="outputs/debug/mesh_overlay.png")
    parser.add_argument("--out-uv-mask", type=str, default="outputs/debug/uv_valid_mask.npy")
    parser.add_argument("--out-uv-pos", type=str, default="outputs/debug/uv_position_map.npy")
    parser.add_argument("--out-uv-nrm", type=str, default="outputs/debug/uv_normal_map.npy")
    parser.add_argument(
        "--transform-mode",
        type=str,
        default="unknown",
        choices=["unknown", "cam2world", "world2cam"],
    )
    args = parser.parse_args()

    dataset = NersembleFastAvatarDataset(camera_dir=args.camera_dir)
    sample = dataset[args.index]

    flame = FlameWrapper(
        flame_model_path=args.flame_model,
        head_template_mesh_path=args.flame_template,
        masks_path=args.flame_masks,
        uv_masks_path=args.uv_masks,
    )
    flame.validate_assets()

    vertices, faces = flame.build_mesh_from_flame_params(sample["flame_params"])
    _ = mesh_vertex_normals(vertices, faces)

    points_uv = project_mesh_vertices(
        vertices_world=vertices,
        intrinsics=sample["intrinsics"],
        transform_matrix=sample["transform_matrix"],
        transform_mode=args.transform_mode,
    )
    vis = points_in_image_mask(points_uv, image_hw=sample["rgb"].shape[:2])

    overlay = _draw_points(sample["rgb"], points_uv, vis)
    _save_image(overlay, Path(args.out_overlay))

    uv_valid = build_uv_valid_mask(uv_resolution=args.uv_resolution)
    uv_pos, uv_nrm = build_geometry_maps_placeholder(uv_resolution=args.uv_resolution)

    Path(args.out_uv_mask).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_uv_mask, uv_valid)
    np.save(args.out_uv_pos, uv_pos)
    np.save(args.out_uv_nrm, uv_nrm)

    print("Saved mesh overlay:", args.out_overlay)
    print("Saved UV placeholder outputs:", args.out_uv_mask, args.out_uv_pos, args.out_uv_nrm)
    print("Visible projected vertices:", int(vis.sum()), "/", int(vis.shape[0]))


if __name__ == "__main__":
    main()
