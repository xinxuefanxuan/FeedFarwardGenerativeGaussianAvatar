"""Select complementary multi-view cameras from subject/cam/clip layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _normalize_timestep_id(x: str | int) -> str:
    s = str(x).strip()
    try:
        return str(int(s))
    except ValueError:
        return s.lstrip("0") or "0"


def _load_frame_for_timestep(transforms_path: Path, timestep_id: str) -> Tuple[Dict[str, object], int]:
    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    target = _normalize_timestep_id(timestep_id)
    for i, frame in enumerate(data["frames"]):
        if _normalize_timestep_id(frame["timestep_id"]) == target:
            return frame, i
    raise KeyError(f"Timestep {timestep_id} not found in {transforms_path}")


def _camera_candidates_from_transform(transform: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Candidate A: transform is cam2world.
    r_a = transform[:3, :3]
    c_a = transform[:3, 3]
    f_a = r_a[:, 2]

    # Candidate B: transform is world2cam -> invert to cam2world.
    inv = np.linalg.inv(transform)
    r_b = inv[:3, :3]
    c_b = inv[:3, 3]
    f_b = r_b[:, 2]

    return [(c_a, f_a), (c_b, f_b)]


def _pick_camera_pose(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    candidates = _camera_candidates_from_transform(transform)
    # Heuristic: pick convention where camera forward roughly points toward world origin.
    scores = []
    for c, f in candidates:
        to_origin = -c
        n = np.linalg.norm(to_origin)
        if n < 1.0e-8:
            scores.append(-1.0)
            continue
        to_origin = to_origin / n
        f = f / (np.linalg.norm(f) + 1.0e-8)
        scores.append(float(np.dot(f, to_origin)))
    return candidates[int(np.argmax(scores))]


def _az_el_from_vec(v: np.ndarray) -> Tuple[float, float, float]:
    r = float(np.linalg.norm(v))
    if r < 1.0e-8:
        return 0.0, 0.0, 0.0
    x, y, z = v / r
    az = float(np.degrees(np.arctan2(x, z)))
    el = float(np.degrees(np.arcsin(np.clip(y, -1.0, 1.0))))
    return az, el, r


def _angular_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1.0e-8)
    b_n = b / (np.linalg.norm(b) + 1.0e-8)
    c = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _greedy_diverse_selection(vectors: List[np.ndarray], num_select: int) -> List[int]:
    n = len(vectors)
    if n == 0:
        return []
    if num_select >= n:
        return list(range(n))
    if n == 1:
        return [0]

    # Initialize with farthest pair.
    best_i, best_j, best_ang = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            ang = _angular_deg(vectors[i], vectors[j])
            if ang > best_ang:
                best_i, best_j, best_ang = i, j, ang
    selected = [best_i]
    if num_select > 1:
        selected.append(best_j)

    while len(selected) < num_select:
        best_k = None
        best_min_ang = -1.0
        for k in range(n):
            if k in selected:
                continue
            min_ang = min(_angular_deg(vectors[k], vectors[s]) for s in selected)
            if min_ang > best_min_ang:
                best_min_ang = min_ang
                best_k = k
        if best_k is None:
            break
        selected.append(best_k)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", type=str, required=True)
    parser.add_argument("--clip-name", type=str, required=True)
    parser.add_argument("--timestep-id", type=str, required=True)
    parser.add_argument("--num-cams", type=int, default=3)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    cams = sorted([p for p in subject_dir.glob("cam_*") if p.is_dir()])
    if not cams:
        raise FileNotFoundError(f"No cam_* directories found under {subject_dir}")

    camera_rows: List[Dict[str, object]] = []
    centers: List[np.ndarray] = []
    for cam_dir in cams:
        transforms_path = cam_dir / args.clip_name / "transforms.json"
        if not transforms_path.exists():
            continue
        try:
            frame, frame_idx = _load_frame_for_timestep(transforms_path, args.timestep_id)
        except KeyError:
            continue
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if transform.shape != (4, 4):
            continue

        center, forward = _pick_camera_pose(transform)
        centers.append(center)
        camera_rows.append(
            {
                "camera_id": cam_dir.name,
                "frame_index": frame_idx,
                "center": center,
                "forward": forward,
            }
        )

    if not camera_rows:
        raise RuntimeError(
            f"No cameras under {subject_dir} had clip={args.clip_name} with timestep={args.timestep_id}"
        )

    centers_np = np.stack(centers, axis=0)
    target = centers_np.mean(axis=0)

    rel_vectors: List[np.ndarray] = []
    for row in camera_rows:
        rel = np.asarray(row["center"], dtype=np.float64) - target
        az, el, r = _az_el_from_vec(rel)
        row["azimuth_deg"] = az
        row["elevation_deg"] = el
        row["radius"] = r
        row["target_center"] = target
        rel_vectors.append(rel)

    selected_idx = _greedy_diverse_selection(rel_vectors, args.num_cams)
    recommended = [camera_rows[i]["camera_id"] for i in selected_idx]

    print("=== Camera orientation summary ===")
    for row in camera_rows:
        print(
            f"{row['camera_id']}: az={row['azimuth_deg']:+7.2f} deg, "
            f"el={row['elevation_deg']:+6.2f} deg, r={row['radius']:.4f}"
        )
    print("=== Recommended cameras (diverse azimuth/elevation) ===")
    print(",".join(recommended))

    result = {
        "subject_dir": str(subject_dir),
        "clip_name": args.clip_name,
        "timestep_id": _normalize_timestep_id(args.timestep_id),
        "num_requested": args.num_cams,
        "recommended_camera_ids": recommended,
        "cameras": [
            {
                "camera_id": str(row["camera_id"]),
                "frame_index": int(row["frame_index"]),
                "center": np.asarray(row["center"], dtype=float).tolist(),
                "forward": np.asarray(row["forward"], dtype=float).tolist(),
                "azimuth_deg": float(row["azimuth_deg"]),
                "elevation_deg": float(row["elevation_deg"]),
                "radius": float(row["radius"]),
            }
            for row in camera_rows
        ],
    }

    out_json = Path(args.out_json) if args.out_json else (subject_dir / f"selected_cameras_{args.clip_name}_{_normalize_timestep_id(args.timestep_id)}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved selection json: {out_json}")
    print("Suggested debug_stage1_multiview.py argument:")
    print(f"--camera-ids {','.join(recommended)}")


if __name__ == "__main__":
    main()
