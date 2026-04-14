"""Debug Stage-1 prior forward and save key visualizations for acceptance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.nersemble_dataset import NersembleFastAvatarDataset
from models.geometry.flame_wrapper import FlameWrapper
from models.stage1_prior.feature_projection import project_image_features_to_surface
from models.stage1_prior.stage1_pipeline import Stage1CanonicalPrior


def _to_chw_float01(arr: np.ndarray) -> torch.Tensor:
    x = np.asarray(arr)
    x = np.squeeze(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        pass
    elif x.ndim == 3 and x.shape[-1] in (1, 3):
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return torch.from_numpy(x)


def _save_gray_png(x: np.ndarray, path: Path) -> None:
    y = np.clip(x, 0.0, 1.0)
    img = (y * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _save_rgb_png(x: np.ndarray, path: Path) -> None:
    y = np.clip(x, 0.0, 1.0)
    img = (y * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def _minmax_chw_to_hwc(x_chw: np.ndarray) -> np.ndarray:
    x = x_chw.astype(np.float32)
    c, _, _ = x.shape
    out = np.zeros_like(x)
    for i in range(c):
        v = x[i]
        v_min = float(v.min())
        v_max = float(v.max())
        if v_max - v_min < 1e-8:
            out[i] = 0.0
        else:
            out[i] = (v - v_min) / (v_max - v_min)
    return np.transpose(out, (1, 2, 0))


def _prepare_batch_from_dataset(
    camera_dir: str,
    start_index: int,
    num_views: int,
    uv_resolution: int,
    flame_model: str,
    flame_template: str,
    flame_masks: str,
    uv_masks: str,
) -> Dict[str, torch.Tensor | str]:
    dataset = NersembleFastAvatarDataset(camera_dir=camera_dir)
    samples: List[Dict[str, object]] = []
    for i in range(num_views):
        samples.append(dataset[start_index + i])

    images = torch.stack([_to_chw_float01(s["rgb"]) for s in samples], dim=0).unsqueeze(0)  # [1,V,C,H,W]

    intrinsics = torch.stack(
        [torch.from_numpy(np.asarray(s["intrinsics"], dtype=np.float32)) for s in samples],
        dim=0,
    ).unsqueeze(0)
    transform_matrices = torch.stack(
        [torch.from_numpy(np.asarray(s["transform_matrix"], dtype=np.float32)) for s in samples],
        dim=0,
    ).unsqueeze(0)

    flame = FlameWrapper(
        flame_model_path=flame_model,
        head_template_mesh_path=flame_template,
        masks_path=flame_masks,
        uv_masks_path=uv_masks,
    )
    mesh_vertices_np, mesh_faces_np = flame.build_mesh_from_flame_params(samples[0]["flame_params"])  # canonical mesh proxy
    mesh_vertices = torch.from_numpy(mesh_vertices_np).float().unsqueeze(0)  # [1,N,3]
    mesh_faces = torch.from_numpy(mesh_faces_np).long()  # [F,3]

    mask_tensors = []
    for s in samples:
        mask = s.get("mask")
        if mask is None:
            mask_tensors.append(None)
            continue
        m = np.asarray(mask).astype(np.float32)
        m = np.squeeze(m)
        if m.ndim == 3:
            m = m[..., 0]
        if m.max() > 1.0:
            m = m / 255.0
        mask_tensors.append(torch.from_numpy(m)[None, None, ...])

    if any(m is None for m in mask_tensors):
        confidence = torch.ones((1, num_views, 1, uv_resolution, uv_resolution), dtype=torch.float32)
    else:
        conf = torch.cat(mask_tensors, dim=0).unsqueeze(0)
        conf = F.interpolate(
            conf.view(num_views, 1, conf.shape[-2], conf.shape[-1]),
            size=(uv_resolution, uv_resolution),
            mode="bilinear",
            align_corners=False,
        ).view(1, num_views, 1, uv_resolution, uv_resolution)
        confidence = conf.clamp(0.0, 1.0)

    return {
        "images": images,
        "confidence": confidence,
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "intrinsics": intrinsics,
        "transform_matrices": transform_matrices,
        "template_mesh_path": flame_template,
    }


def _prepare_batch_random(batch_size: int, num_views: int, h: int, w: int, uv_resolution: int) -> Dict[str, torch.Tensor]:
    images = torch.rand(batch_size, num_views, 3, h, w)
    confidence = torch.rand(batch_size, num_views, 1, uv_resolution, uv_resolution)

    # simple plane mesh with UV-like structure for smoke test
    yy, xx = torch.meshgrid(torch.linspace(-0.2, 0.2, 64), torch.linspace(-0.2, 0.2, 64), indexing="ij")
    zz = torch.ones_like(xx) * 1.0
    verts = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

    faces = []
    for y in range(63):
        for x in range(63):
            i = y * 64 + x
            faces.append([i, i + 1, i + 64])
            faces.append([i + 1, i + 65, i + 64])
    faces = torch.tensor(faces, dtype=torch.long)

    intr = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    intr[..., 0, 0] = 500.0
    intr[..., 1, 1] = 500.0
    intr[..., 0, 2] = w / 2.0
    intr[..., 1, 2] = h / 2.0

    t = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)

    uv_verts = torch.stack([
        (verts[:, 0] - verts[:, 0].min()) / (verts[:, 0].max() - verts[:, 0].min() + 1e-8),
        (verts[:, 1] - verts[:, 1].min()) / (verts[:, 1].max() - verts[:, 1].min() + 1e-8),
    ], dim=-1)

    return {
        "images": images,
        "confidence": confidence,
        "mesh_vertices": verts.unsqueeze(0).repeat(batch_size, 1, 1),
        "mesh_faces": faces,
        "uv_vertices": uv_verts,
        "uv_faces": faces,
        "intrinsics": intr,
        "transform_matrices": t,
        "template_mesh_path": "/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-dir", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--uv-resolution", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="outputs/stage1_debug")
    parser.add_argument("--use-random", action="store_true")
    parser.add_argument("--flame-model", type=str, default="/home/yuanyuhao/VHAP/asset/flame/flame2023.pkl")
    parser.add_argument("--flame-template", type=str, default="/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj")
    parser.add_argument("--flame-masks", type=str, default="/home/yuanyuhao/VHAP/asset/flame/FLAME_masks.pkl")
    parser.add_argument("--uv-masks", type=str, default="/home/yuanyuhao/VHAP/asset/flame/uv_masks.npz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_random:
        batch = _prepare_batch_random(batch_size=1, num_views=args.num_views, h=512, w=512, uv_resolution=args.uv_resolution)
    else:
        if args.camera_dir is None:
            raise ValueError("`--camera-dir` is required unless --use-random is set")
        batch = _prepare_batch_from_dataset(
            camera_dir=args.camera_dir,
            start_index=args.start_index,
            num_views=args.num_views,
            uv_resolution=args.uv_resolution,
            flame_model=args.flame_model,
            flame_template=args.flame_template,
            flame_masks=args.flame_masks,
            uv_masks=args.uv_masks,
        )

    model = Stage1CanonicalPrior(uv_resolution=args.uv_resolution)
    model.eval()

    with torch.no_grad():
        outputs = model(batch)
        projection_debug = {}
        if all(
            k in outputs
            for k in (
                "uv_feature_single_raw",
                "uv_feature_single_masked",
                "uv_feature_single_weighted",
                "uv_feature_single_vflip",
                "uv_visibility_single",
                "uv_confidence_single",
                "uv_face_index",
                "uv_barycentric_vis",
                "uv_sample_xy",
            )
        ):
            projection_debug = {
                "uv_feature_single_raw": outputs["uv_feature_single_raw"],
                "uv_feature_single_masked": outputs["uv_feature_single_masked"],
                "uv_feature_single_weighted": outputs["uv_feature_single_weighted"],
                "uv_feature_single_vflip": outputs["uv_feature_single_vflip"],
                "uv_visibility_single": outputs["uv_visibility_single"],
                "uv_confidence_single": outputs["uv_confidence_single"],
                "uv_face_index": outputs["uv_face_index"],
                "uv_barycentric_vis": outputs["uv_barycentric_vis"],
                "uv_sample_xy": outputs["uv_sample_xy"],
            }
        else:
            image_features = model.builder.encoder(batch["images"])
            projection_debug = project_image_features_to_surface(
                image_features=image_features,
                mesh_vertices=batch["mesh_vertices"],
                mesh_faces=batch["mesh_faces"],
                intrinsics=batch["intrinsics"],
                transform_matrices=batch["transform_matrices"],
                template_mesh_path=batch.get("template_mesh_path", model.builder.template_mesh_path),
                uv_resolution=args.uv_resolution,
                uv_vertices=batch.get("uv_vertices"),
                uv_faces=batch.get("uv_faces"),
            )
            projection_debug["uv_feature_single_raw"] = projection_debug["uv_feature_single"]
            projection_debug["uv_feature_single_masked"] = (
                projection_debug["uv_feature_single"] * projection_debug["uv_visibility_single"]
            )
            projection_debug["uv_confidence_single"] = batch["confidence"][:, 0]
            projection_debug["uv_feature_single_weighted"] = (
                projection_debug["uv_feature_single_masked"] * projection_debug["uv_confidence_single"]
            )

    uv_valid_mask = outputs["uv_valid_mask"][0, 0].detach().cpu().numpy()
    uv_position_map = outputs["uv_position_map"][0].detach().cpu().numpy()
    uv_normal_map = outputs["uv_normal_map"][0].detach().cpu().numpy()
    fused_uv = outputs["fused_uv_feature_map"][0].detach().cpu().numpy()
    fusion_weights = outputs.get("fusion_weights")

    np.save(out_dir / "uv_position_map.npy", uv_position_map)
    np.save(out_dir / "uv_normal_map.npy", uv_normal_map)

    _save_gray_png(uv_valid_mask, out_dir / "uv_valid_mask.png")
    _save_rgb_png(_minmax_chw_to_hwc(uv_position_map), out_dir / "uv_position_map.png")
    _save_rgb_png(np.transpose((uv_normal_map + 1.0) * 0.5, (1, 2, 0)), out_dir / "uv_normal_map.png")

    if fused_uv.shape[0] >= 3:
        fused_vis = _minmax_chw_to_hwc(fused_uv[:3])
    else:
        pad = np.zeros((3, fused_uv.shape[1], fused_uv.shape[2]), dtype=np.float32)
        pad[: fused_uv.shape[0]] = fused_uv
        fused_vis = _minmax_chw_to_hwc(pad)
    _save_rgb_png(fused_vis, out_dir / "fused_uv_feature_map.png")

    if fusion_weights is not None:
        fw = fusion_weights[0].detach().cpu().numpy()  # [V,1,H,W]
        fw_mean = fw.mean(axis=0)[0]
        fw_vis = fw_mean / max(float(fw_mean.max()), 1.0e-8)
        _save_gray_png(fw_vis, out_dir / "fusion_weights_mean.png")

    uv_feature_single_raw = projection_debug["uv_feature_single_raw"][0].detach().cpu().numpy()
    if uv_feature_single_raw.shape[0] >= 3:
        uv_feature_single_raw_vis = _minmax_chw_to_hwc(uv_feature_single_raw[:3])
    else:
        pad = np.zeros((3, uv_feature_single_raw.shape[1], uv_feature_single_raw.shape[2]), dtype=np.float32)
        pad[: uv_feature_single_raw.shape[0]] = uv_feature_single_raw
        uv_feature_single_raw_vis = _minmax_chw_to_hwc(pad)
    _save_rgb_png(uv_feature_single_raw_vis, out_dir / "uv_feature_single_raw.png")
    _save_rgb_png(uv_feature_single_raw_vis, out_dir / "uv_feature_single.png")

    uv_feature_single_masked = projection_debug["uv_feature_single_masked"][0].detach().cpu().numpy()
    if uv_feature_single_masked.shape[0] >= 3:
        uv_feature_single_masked_vis = _minmax_chw_to_hwc(uv_feature_single_masked[:3])
    else:
        pad = np.zeros((3, uv_feature_single_masked.shape[1], uv_feature_single_masked.shape[2]), dtype=np.float32)
        pad[: uv_feature_single_masked.shape[0]] = uv_feature_single_masked
        uv_feature_single_masked_vis = _minmax_chw_to_hwc(pad)
    _save_rgb_png(uv_feature_single_masked_vis, out_dir / "uv_feature_single_masked.png")

    uv_feature_single_weighted = projection_debug["uv_feature_single_weighted"][0].detach().cpu().numpy()
    if uv_feature_single_weighted.shape[0] >= 3:
        uv_feature_single_weighted_vis = _minmax_chw_to_hwc(uv_feature_single_weighted[:3])
    else:
        pad = np.zeros((3, uv_feature_single_weighted.shape[1], uv_feature_single_weighted.shape[2]), dtype=np.float32)
        pad[: uv_feature_single_weighted.shape[0]] = uv_feature_single_weighted
        uv_feature_single_weighted_vis = _minmax_chw_to_hwc(pad)
    _save_rgb_png(uv_feature_single_weighted_vis, out_dir / "uv_feature_single_weighted.png")

    uv_feature_single_vflip = projection_debug["uv_feature_single_vflip"][0].detach().cpu().numpy()
    if uv_feature_single_vflip.shape[0] >= 3:
        uv_feature_single_vflip_vis = _minmax_chw_to_hwc(uv_feature_single_vflip[:3])
    else:
        pad = np.zeros((3, uv_feature_single_vflip.shape[1], uv_feature_single_vflip.shape[2]), dtype=np.float32)
        pad[: uv_feature_single_vflip.shape[0]] = uv_feature_single_vflip
        uv_feature_single_vflip_vis = _minmax_chw_to_hwc(pad)
    _save_rgb_png(uv_feature_single_vflip_vis, out_dir / "uv_feature_single_vflip.png")

    uv_visibility_single = projection_debug["uv_visibility_single"][0, 0].detach().cpu().numpy()
    uv_visibility_single = uv_visibility_single / max(float(uv_visibility_single.max()), 1.0e-8)
    _save_gray_png(uv_visibility_single, out_dir / "uv_visibility_single.png")

    uv_confidence_single = projection_debug["uv_confidence_single"][0, 0].detach().cpu().numpy()
    uv_confidence_single = np.clip(uv_confidence_single, 0.0, 1.0)
    _save_gray_png(uv_confidence_single, out_dir / "uv_confidence_single.png")

    uv_face_index = projection_debug["uv_face_index"][0, 0].detach().cpu().numpy()
    uv_face_index = uv_face_index - float(uv_face_index.min())
    uv_face_index = uv_face_index / max(float(uv_face_index.max()), 1.0e-8)
    _save_gray_png(uv_face_index, out_dir / "uv_face_index.png")

    uv_barycentric_vis = projection_debug["uv_barycentric_vis"][0].detach().cpu().numpy()
    _save_rgb_png(_minmax_chw_to_hwc(uv_barycentric_vis), out_dir / "uv_barycentric_vis.png")

    uv_sample_xy = projection_debug["uv_sample_xy"][0].detach().cpu().numpy()
    np.save(out_dir / "uv_sample_xy.npy", uv_sample_xy)
    sample_x = uv_sample_xy[0]
    sample_y = uv_sample_xy[1]
    img_h = float(batch["images"].shape[-2])
    img_w = float(batch["images"].shape[-1])
    sample_xn = np.clip(sample_x / max(img_w - 1.0, 1.0), 0.0, 1.0)
    sample_yn = np.clip(sample_y / max(img_h - 1.0, 1.0), 0.0, 1.0)
    sample_mask = np.clip(projection_debug["uv_visibility_single"][0, 0].detach().cpu().numpy(), 0.0, 1.0)
    _save_rgb_png(np.stack([sample_xn, sample_yn, sample_mask], axis=-1), out_dir / "uv_sample_xy.png")

    input_rgb = batch["images"][0, 0, :3].detach().cpu().numpy()
    reproject_vis = np.transpose(np.clip(input_rgb, 0.0, 1.0), (1, 2, 0))
    xy = uv_sample_xy.reshape(2, -1).T
    vis = sample_mask.reshape(-1) > 0.5
    if vis.any():
        xy_valid = xy[vis]
        stride = max(1, xy_valid.shape[0] // 15000)
        xy_plot = np.round(xy_valid[::stride]).astype(np.int32)
        x = np.clip(xy_plot[:, 0], 0, reproject_vis.shape[1] - 1)
        y = np.clip(xy_plot[:, 1], 0, reproject_vis.shape[0] - 1)
        reproject_vis[y, x, 0] = 1.0
        reproject_vis[y, x, 1] = 0.0
        reproject_vis[y, x, 2] = 0.0
    _save_rgb_png(reproject_vis, out_dir / "reprojected_points_on_image.png")

    vis_ratio = float((uv_visibility_single > 0.5).mean())
    raw_nonzero_ratio = float((np.abs(uv_feature_single_raw) > 1.0e-8).mean())
    masked_nonzero_ratio = float((np.abs(uv_feature_single_masked) > 1.0e-8).mean())
    fused_uv_map = outputs["fused_uv_feature_map"][0].detach().cpu().numpy()
    fused_masked_l1 = float(np.mean(np.abs(fused_uv_map - uv_feature_single_masked)))
    fused_masked_l2 = float(np.sqrt(np.mean((fused_uv_map - uv_feature_single_masked) ** 2)))
    conf_mean = float(uv_confidence_single.mean())
    conf_nonzero_ratio = float((uv_confidence_single > 1.0e-8).mean())
    weighted_fused_l1 = float(np.mean(np.abs(fused_uv_map - uv_feature_single_weighted)))
    weighted_fused_l2 = float(np.sqrt(np.mean((fused_uv_map - uv_feature_single_weighted) ** 2)))
    print(f"[single-view-debug] uv_visibility_single valid ratio: {vis_ratio:.6f}")
    print(f"[single-view-debug] uv_confidence_single mean: {conf_mean:.6f}")
    print(f"[single-view-debug] uv_confidence_single nonzero ratio: {conf_nonzero_ratio:.6f}")
    print(f"[single-view-debug] uv_feature_single_raw nonzero ratio: {raw_nonzero_ratio:.6f}")
    print(f"[single-view-debug] uv_feature_single_masked nonzero ratio: {masked_nonzero_ratio:.6f}")
    print(f"[single-view-debug] fused_vs_masked L1: {fused_masked_l1:.6f}")
    print(f"[single-view-debug] fused_vs_masked L2: {fused_masked_l2:.6f}")
    print(f"[single-view-debug] weighted_vs_fused L1: {weighted_fused_l1:.6f}")
    print(f"[single-view-debug] weighted_vs_fused L2: {weighted_fused_l2:.6f}")

    print("Saved Stage1 debug outputs to:", out_dir)


if __name__ == "__main__":
    main()
