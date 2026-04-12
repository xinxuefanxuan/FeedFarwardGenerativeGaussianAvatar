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
) -> Dict[str, torch.Tensor]:
    dataset = NersembleFastAvatarDataset(camera_dir=camera_dir)
    samples: List[Dict[str, object]] = []
    for i in range(num_views):
        samples.append(dataset[start_index + i])

    images = torch.stack([_to_chw_float01(s["rgb"]) for s in samples], dim=0).unsqueeze(0)

    # Build visibility/confidence from masks if available; fallback to ones.
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
        conf = torch.cat(mask_tensors, dim=0).unsqueeze(0)  # [1,V,1,H,W]
        conf = F.interpolate(
            conf.view(num_views, 1, conf.shape[-2], conf.shape[-1]),
            size=(uv_resolution, uv_resolution),
            mode="bilinear",
            align_corners=False,
        ).view(1, num_views, 1, uv_resolution, uv_resolution)
        confidence = conf.clamp(0.0, 1.0)

    visibility = (confidence > 0.1).float()
    return {
        "images": images,
        "confidence": confidence,
        "visibility": visibility,
    }


def _prepare_batch_random(batch_size: int, num_views: int, h: int, w: int, uv_resolution: int) -> Dict[str, torch.Tensor]:
    images = torch.rand(batch_size, num_views, 3, h, w)
    confidence = torch.rand(batch_size, num_views, 1, uv_resolution, uv_resolution)
    visibility = (confidence > 0.2).float()
    return {"images": images, "confidence": confidence, "visibility": visibility}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-dir", type=str, default=None, help="FastAvatar processed camera directory")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--uv-resolution", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="outputs/stage1_debug")
    parser.add_argument("--use-random", action="store_true", help="Use random synthetic inputs instead of dataset")
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
        )

    model = Stage1CanonicalPrior(uv_resolution=args.uv_resolution)
    model.eval()

    with torch.no_grad():
        outputs = model(batch)

    # Tensors are [B,...], save first sample.
    uv_valid_mask = outputs["uv_valid_mask"][0, 0].detach().cpu().numpy()
    uv_position_map = outputs["uv_position_map"][0].detach().cpu().numpy()  # [3,H,W]
    uv_normal_map = outputs["uv_normal_map"][0].detach().cpu().numpy()  # [3,H,W]
    fused_uv = outputs["fused_uv_feature_map"][0].detach().cpu().numpy()  # [C,H,W]
    fusion_weights = outputs.get("fusion_weights")

    # save npy
    np.save(out_dir / "uv_position_map.npy", uv_position_map)
    np.save(out_dir / "uv_normal_map.npy", uv_normal_map)

    # save png
    _save_gray_png(uv_valid_mask, out_dir / "uv_valid_mask.png")

    uv_pos_vis = _minmax_chw_to_hwc(uv_position_map)
    _save_rgb_png(uv_pos_vis, out_dir / "uv_position_map.png")

    uv_nrm_vis = np.transpose((uv_normal_map + 1.0) * 0.5, (1, 2, 0))
    _save_rgb_png(uv_nrm_vis, out_dir / "uv_normal_map.png")

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
        fw_vis = fw_mean
        if fw_vis.max() > 1e-8:
            fw_vis = fw_vis / fw_vis.max()
        _save_gray_png(fw_vis, out_dir / "fusion_weights_mean.png")

    print("Saved Stage1 debug outputs to:", out_dir)
    print(" - uv_valid_mask.png")
    print(" - uv_position_map.png + uv_position_map.npy")
    print(" - uv_normal_map.png + uv_normal_map.npy")
    print(" - fused_uv_feature_map.png")
    if fusion_weights is not None:
        print(" - fusion_weights_mean.png")


if __name__ == "__main__":
    main()
