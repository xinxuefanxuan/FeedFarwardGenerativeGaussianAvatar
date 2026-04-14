"""Build a true multi-view Stage-1 debug batch from subject/cam/clip layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
        if v_max - v_min < 1.0e-8:
            out[i] = 0.0
        else:
            out[i] = (v - v_min) / (v_max - v_min)
    return np.transpose(out, (1, 2, 0))


def _normalize_timestep_id(x: str | int) -> str:
    s = str(x).strip()
    try:
        return str(int(s))
    except ValueError:
        return s.lstrip("0") or "0"


def _find_frame_index_by_timestep(dataset: NersembleFastAvatarDataset, timestep_id: str) -> int | None:
    target = _normalize_timestep_id(timestep_id)
    for i, frame in enumerate(dataset.frames):
        if _normalize_timestep_id(frame["timestep_id"]) == target:
            return i
    return None


def _parse_camera_ids(camera_ids_arg: str | None) -> List[str] | None:
    if camera_ids_arg is None or camera_ids_arg.strip() == "":
        return None
    return [x.strip() for x in camera_ids_arg.split(",") if x.strip()]


def _discover_cameras(subject_dir: Path, clip_name: str) -> List[str]:
    cams = []
    for p in sorted(subject_dir.glob("cam_*")):
        if not p.is_dir():
            continue
        if (p / clip_name / "transforms.json").exists():
            cams.append(p.name)
    return cams


def _prepare_multiview_batch(
    subject_dir: Path,
    clip_name: str,
    timestep_id: str,
    num_cams: int,
    camera_ids: Sequence[str] | None,
    uv_resolution: int,
    flame_model: str,
    flame_template: str,
    flame_masks: str,
    uv_masks: str,
) -> Dict[str, object]:
    available_cams = _discover_cameras(subject_dir, clip_name)
    if not available_cams:
        raise FileNotFoundError(f"No usable cameras found under {subject_dir} for clip {clip_name}")

    if camera_ids is not None:
        requested = [c for c in camera_ids if c in available_cams]
        missing = [c for c in camera_ids if c not in available_cams]
        if missing:
            raise FileNotFoundError(f"Requested cameras not found for clip {clip_name}: {missing}")
        target_num = len(requested)
    else:
        requested = available_cams
        target_num = num_cams

    selected_samples: List[Dict[str, object]] = []
    selected_cam_ids: List[str] = []
    failed_cameras: List[Tuple[str, str]] = []
    for cam_id in requested:
        cam_clip_dir = subject_dir / cam_id / clip_name
        try:
            ds = NersembleFastAvatarDataset(camera_dir=cam_clip_dir)
            frame_idx = _find_frame_index_by_timestep(ds, timestep_id)
            if frame_idx is None:
                failed_cameras.append((cam_id, f"timestep {timestep_id} not found"))
                continue
            selected_samples.append(ds[frame_idx])
            selected_cam_ids.append(cam_id)
        except Exception as exc:  # debug tool should report per-camera failure details
            failed_cameras.append((cam_id, str(exc)))
            continue
        if camera_ids is None and len(selected_samples) >= target_num:
            break

    if len(selected_samples) < target_num:
        raise RuntimeError(
            f"Found only {len(selected_samples)} cameras with timestep {timestep_id}, required {target_num}. "
            f"Requested={requested}, usable cameras={available_cams}, failures={failed_cameras}"
        )

    images = torch.stack([_to_chw_float01(s["rgb"]) for s in selected_samples], dim=0).unsqueeze(0)
    intrinsics = torch.stack(
        [torch.from_numpy(np.asarray(s["intrinsics"], dtype=np.float32)) for s in selected_samples],
        dim=0,
    ).unsqueeze(0)
    transform_matrices = torch.stack(
        [torch.from_numpy(np.asarray(s["transform_matrix"], dtype=np.float32)) for s in selected_samples],
        dim=0,
    ).unsqueeze(0)

    flame = FlameWrapper(
        flame_model_path=flame_model,
        head_template_mesh_path=flame_template,
        masks_path=flame_masks,
        uv_masks_path=uv_masks,
    )
    mesh_vertices_np, mesh_faces_np = flame.build_mesh_from_flame_params(selected_samples[0]["flame_params"])
    mesh_vertices = torch.from_numpy(mesh_vertices_np).float().unsqueeze(0)
    mesh_faces = torch.from_numpy(mesh_faces_np).long()

    masks = []
    for s in selected_samples:
        m = s.get("mask")
        if m is None:
            masks.append(None)
            continue
        mask = np.asarray(m).astype(np.float32)
        mask = np.squeeze(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.max() > 1.0:
            mask = mask / 255.0
        masks.append(torch.from_numpy(mask)[None, None, ...])

    if any(m is None for m in masks):
        confidence = torch.ones((1, len(selected_samples), 1, uv_resolution, uv_resolution), dtype=torch.float32)
    else:
        conf = torch.cat(masks, dim=0).unsqueeze(0)
        conf = F.interpolate(
            conf.view(len(selected_samples), 1, conf.shape[-2], conf.shape[-1]),
            size=(uv_resolution, uv_resolution),
            mode="bilinear",
            align_corners=False,
        ).view(1, len(selected_samples), 1, uv_resolution, uv_resolution)
        confidence = conf.clamp(0.0, 1.0)

    return {
        "batch": {
            "images": images,
            "confidence": confidence,
            "mesh_vertices": mesh_vertices,
            "mesh_faces": mesh_faces,
            "intrinsics": intrinsics,
            "transform_matrices": transform_matrices,
            "template_mesh_path": flame_template,
        },
        "camera_ids": selected_cam_ids,
        "requested_camera_ids": list(requested),
        "failed_cameras": failed_cameras,
        "timestep_id": _normalize_timestep_id(timestep_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", type=str, required=True)
    parser.add_argument("--clip-name", type=str, required=True)
    parser.add_argument("--timestep-id", type=str, required=True)
    parser.add_argument("--num-cams", type=int, default=3)
    parser.add_argument("--camera-ids", type=str, default=None, help="Comma-separated camera ids, e.g. cam_1,cam_2")
    parser.add_argument("--uv-resolution", type=int, default=256)
    parser.add_argument("--flame-model", type=str, default="/home/yuanyuhao/VHAP/asset/flame/flame2023.pkl")
    parser.add_argument("--flame-template", type=str, default="/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj")
    parser.add_argument("--flame-masks", type=str, default="/home/yuanyuhao/VHAP/asset/flame/FLAME_masks.pkl")
    parser.add_argument("--uv-masks", type=str, default="/home/yuanyuhao/VHAP/asset/flame/uv_masks.npz")
    parser.add_argument("--out-dir", type=str, default="outputs/stage1_multiview_debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = _prepare_multiview_batch(
        subject_dir=Path(args.subject_dir),
        clip_name=args.clip_name,
        timestep_id=args.timestep_id,
        num_cams=args.num_cams,
        camera_ids=_parse_camera_ids(args.camera_ids),
        uv_resolution=args.uv_resolution,
        flame_model=args.flame_model,
        flame_template=args.flame_template,
        flame_masks=args.flame_masks,
        uv_masks=args.uv_masks,
    )
    batch = prep["batch"]
    camera_ids = prep["camera_ids"]
    requested_camera_ids = prep["requested_camera_ids"]
    failed_cameras = prep["failed_cameras"]
    timestep_id = prep["timestep_id"]

    print(f"[multiview-debug] requested camera_ids={requested_camera_ids}")
    if failed_cameras:
        for cam_id, reason in failed_cameras:
            print(f"[multiview-debug] failed camera={cam_id}, reason={reason}")
    print(f"[multiview-debug] actually loaded camera_ids={camera_ids}")
    print(f"[multiview-debug] final batch view count={len(camera_ids)}")

    model = Stage1CanonicalPrior(uv_resolution=args.uv_resolution)
    model.eval()

    with torch.no_grad():
        outputs = model(batch)
        image_features = model.builder.encoder(batch["images"])
        projection_debug = project_image_features_to_surface(
            image_features=image_features,
            mesh_vertices=batch["mesh_vertices"],
            mesh_faces=batch["mesh_faces"],
            intrinsics=batch["intrinsics"],
            transform_matrices=batch["transform_matrices"],
            template_mesh_path=batch["template_mesh_path"],
            uv_resolution=args.uv_resolution,
            uv_vertices=batch.get("uv_vertices"),
            uv_faces=batch.get("uv_faces"),
        )

    uv_valid_mask = outputs["uv_valid_mask"][0, 0].detach().cpu().numpy()
    uv_position_map = outputs["uv_position_map"][0].detach().cpu().numpy()
    uv_normal_map = outputs["uv_normal_map"][0].detach().cpu().numpy()
    fused_uv = outputs["fused_uv_feature_map"][0].detach().cpu().numpy()
    fusion_weights = outputs.get("fusion_weights")

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
        fw = fusion_weights[0].detach().cpu().numpy()
        fw_mean = fw.mean(axis=0)[0]
        fw_vis = fw_mean / max(float(fw_mean.max()), 1.0e-8)
        _save_gray_png(fw_vis, out_dir / "fusion_weights_mean.png")

    uv_features = projection_debug["uv_features"][0].detach().cpu().numpy()
    uv_visibility = projection_debug["uv_visibility"][0].detach().cpu().numpy()
    for vi in range(uv_features.shape[0]):
        view_feat = uv_features[vi]
        if view_feat.shape[0] >= 3:
            view_vis = _minmax_chw_to_hwc(view_feat[:3])
        else:
            pad = np.zeros((3, view_feat.shape[1], view_feat.shape[2]), dtype=np.float32)
            pad[: view_feat.shape[0]] = view_feat
            view_vis = _minmax_chw_to_hwc(pad)
        _save_rgb_png(view_vis, out_dir / f"uv_feature_view{vi}.png")

        vis_ratio = float((uv_visibility[vi, 0] > 0.5).mean())
        print(f"[multiview-debug] view={vi} cam={camera_ids[vi]} visible_ratio={vis_ratio:.6f}")

    fused_nonzero_ratio = float((np.abs(fused_uv) > 1.0e-8).mean())
    print(f"[multiview-debug] cameras={camera_ids}")
    print(f"[multiview-debug] timestep_id={timestep_id}")
    print(f"[multiview-debug] fused_nonzero_ratio={fused_nonzero_ratio:.6f}")
    print(f"Saved Stage1 multi-view debug outputs to: {out_dir}")


if __name__ == "__main__":
    main()
