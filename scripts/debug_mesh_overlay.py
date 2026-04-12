"""Debug FLAME mesh overlay on a real sample image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from datasets.nersemble_dataset import NersembleFastAvatarDataset
from models.geometry.flame_wrapper import FlameWrapper
from models.geometry.mesh_ops import mesh_vertex_normals
from models.geometry.uv_ops import build_geometry_maps_placeholder, build_uv_valid_mask
from models.geometry.visibility import points_in_image_mask


def _to_hwc_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.floating) and (float(np.max(arr)) if arr.size else 0.0) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _stats(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    arr_min = np.min(arr, axis=0)
    arr_max = np.max(arr, axis=0)
    arr_range = arr_max - arr_min
    return f"{name}: min={arr_min}, max={arr_max}, range={arr_range}"


def _camera_transform(vertices_world: np.ndarray, transform_matrix: np.ndarray, mode: str) -> np.ndarray:
    verts_h = np.concatenate([vertices_world, np.ones((vertices_world.shape[0], 1), dtype=vertices_world.dtype)], axis=1)
    if mode == "none":
        cam = verts_h[:, :3]
    elif mode == "world2cam":
        cam = (transform_matrix @ verts_h.T).T[:, :3]
    elif mode == "cam2world":
        world2cam = np.linalg.inv(transform_matrix)
        cam = (world2cam @ verts_h.T).T[:, :3]
    else:
        raise ValueError(f"Unsupported projection mode: {mode}")
    return cam


def _project_from_camera(vertices_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(vertices_cam[:, 2:3], 1e-6, None)
    x = vertices_cam[:, 0:1] / z
    y = vertices_cam[:, 1:2] / z
    uv_h = (intrinsics @ np.concatenate([x, y, np.ones_like(x)], axis=1).T).T
    return uv_h[:, :2]


def _run_projection_chain(
    name: str,
    vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    transform_matrix: np.ndarray,
    image_hw: Tuple[int, int],
    debug: bool,
) -> Dict[str, np.ndarray | int]:
    vertices_cam = _camera_transform(vertices_world, transform_matrix, mode=name)
    points_uv = _project_from_camera(vertices_cam, intrinsics)
    vis = points_in_image_mask(points_uv, image_hw=image_hw)

    if debug:
        print(f"\n[debug-projection] chain={name}")
        print(_stats("vertices_cam", vertices_cam))
        z = vertices_cam[:, 2]
        print(f"z>0: {int((z > 0).sum())} / {z.shape[0]}")
        print(f"z<0: {int((z < 0).sum())} / {z.shape[0]}")
        print(_stats("projected_xy", points_uv))
        print(f"inside_image: {int(vis.sum())} / {vis.shape[0]}")

    return {
        "points_uv": points_uv,
        "visible": int(vis.sum()),
        "vis_mask": vis,
    }


def _draw_overlay(rgb: np.ndarray, points_uv: np.ndarray, valid_mask: np.ndarray, faces: np.ndarray) -> np.ndarray:
    canvas = _to_hwc_uint8(rgb)

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to save debug images") from exc

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    # Draw wireframe edges for visible triangles.
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid_mask[i0] and valid_mask[i1] and valid_mask[i2]):
            continue
        p0 = tuple(points_uv[i0].tolist())
        p1 = tuple(points_uv[i1].tolist())
        p2 = tuple(points_uv[i2].tolist())
        draw.line([p0, p1], fill=(0, 255, 0), width=1)
        draw.line([p1, p2], fill=(0, 255, 0), width=1)
        draw.line([p2, p0], fill=(0, 255, 0), width=1)

    # Draw projected vertices scatter.
    for (u, v), is_valid in zip(points_uv, valid_mask):
        if not is_valid:
            continue
        r = 1
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 0, 0))

    return np.asarray(img, dtype=np.uint8)


def _save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save debug images") from exc
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
        default="world2cam",
        choices=["none", "cam2world", "world2cam"],
        help="Which projection chain to use for final overlay output.",
    )
    parser.add_argument(
        "--debug-projection",
        action="store_true",
        help="Print detailed projection statistics for none/world2cam/cam2world chains.",
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

    if args.debug_projection:
        print("[debug-projection]", _stats("vertices_world", vertices))
        translation = sample["flame_params"].get("translation")
        if translation is not None:
            translation = np.asarray(translation).reshape(-1)
            print(f"[debug-projection] flame translation: {translation[:3]}")
        else:
            print("[debug-projection] flame translation: None")

    image_hw = _to_hwc_uint8(sample["rgb"]).shape[:2]
    chain_results: Dict[str, Dict[str, np.ndarray | int]] = {}
    for chain in ("none", "world2cam", "cam2world"):
        chain_results[chain] = _run_projection_chain(
            name=chain,
            vertices_world=vertices,
            intrinsics=sample["intrinsics"],
            transform_matrix=sample["transform_matrix"],
            image_hw=image_hw,
            debug=args.debug_projection,
        )

    for chain in ("none", "world2cam", "cam2world"):
        print(f"Visible projected vertices ({chain}): {chain_results[chain]['visible']} / {vertices.shape[0]}")

    selected = chain_results[args.transform_mode]
    overlay = _draw_overlay(sample["rgb"], selected["points_uv"], selected["vis_mask"], faces)
    _save_image(overlay, Path(args.out_overlay))

    uv_valid = build_uv_valid_mask(uv_resolution=args.uv_resolution)
    uv_pos, uv_nrm = build_geometry_maps_placeholder(uv_resolution=args.uv_resolution)

    Path(args.out_uv_mask).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_uv_mask, uv_valid)
    np.save(args.out_uv_pos, uv_pos)
    np.save(args.out_uv_nrm, uv_nrm)

    print("Saved mesh overlay:", args.out_overlay)
    print("Saved UV placeholder outputs:", args.out_uv_mask, args.out_uv_pos, args.out_uv_nrm)


if __name__ == "__main__":
    main()
