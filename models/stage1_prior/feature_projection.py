"""Real image->surface->UV projection for Stage 1 MVP."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class UVTemplate:
    uv_vertices: torch.Tensor
    uv_faces: torch.Tensor
    mesh_faces_from_obj: torch.Tensor


def _parse_obj_with_uv(obj_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv_vertices = []
    uv_faces = []
    mesh_faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("vt "):
                _, u, v = line.strip().split()[:3]
                uv_vertices.append([float(u), float(v)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                vidx = []
                vt_idx = []
                for p in parts:
                    comps = p.split("/")
                    if len(comps) >= 1 and comps[0] != "":
                        vidx.append(int(comps[0]) - 1)
                    if len(comps) >= 2 and comps[1] != "":
                        vt_idx.append(int(comps[1]) - 1)
                if len(vidx) == 3 and len(vt_idx) == 3:
                    mesh_faces.append(vidx)
                    uv_faces.append(vt_idx)
                elif len(vidx) > 3 and len(vt_idx) > 3:
                    for i in range(1, len(vidx) - 1):
                        mesh_faces.append([vidx[0], vidx[i], vidx[i + 1]])
                        uv_faces.append([vt_idx[0], vt_idx[i], vt_idx[i + 1]])

    if len(uv_vertices) == 0 or len(uv_faces) == 0 or len(mesh_faces) == 0:
        raise ValueError(f"OBJ missing UV/face data: {obj_path}")

    return (
        np.asarray(uv_vertices, dtype=np.float32),
        np.asarray(uv_faces, dtype=np.int64),
        np.asarray(mesh_faces, dtype=np.int64),
    )


@lru_cache(maxsize=4)
def load_uv_template(obj_path: str) -> UVTemplate:
    uv_vertices, uv_faces, mesh_faces = _parse_obj_with_uv(Path(obj_path))
    return UVTemplate(
        uv_vertices=torch.from_numpy(uv_vertices),
        uv_faces=torch.from_numpy(uv_faces),
        mesh_faces_from_obj=torch.from_numpy(mesh_faces),
    )


def _rasterize_uv(
    uv_vertices: torch.Tensor,
    uv_faces: torch.Tensor,
    uv_resolution: int,
    v_flip: bool,
) -> Dict[str, torch.Tensor]:
    device = uv_vertices.device
    dtype = uv_vertices.dtype
    h = uv_resolution
    w = uv_resolution
    tri_idx = torch.full((h, w), -1, device=device, dtype=torch.long)
    bary = torch.zeros((h, w, 3), device=device, dtype=dtype)

    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w)
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(h)

    for fi in range(uv_faces.shape[0]):
        i0, i1, i2 = uv_faces[fi]
        p0 = uv_vertices[i0]
        p1 = uv_vertices[i1]
        p2 = uv_vertices[i2]

        min_u = torch.clamp(torch.min(torch.stack([p0[0], p1[0], p2[0]])), 0.0, 1.0)
        max_u = torch.clamp(torch.max(torch.stack([p0[0], p1[0], p2[0]])), 0.0, 1.0)
        min_v = torch.clamp(torch.min(torch.stack([p0[1], p1[1], p2[1]])), 0.0, 1.0)
        max_v = torch.clamp(torch.max(torch.stack([p0[1], p1[1], p2[1]])), 0.0, 1.0)

        x0 = int(torch.floor(min_u * w).item())
        x1 = int(torch.ceil(max_u * w).item())
        if v_flip:
            y0 = int(torch.floor((1.0 - max_v) * h).item())
            y1 = int(torch.ceil((1.0 - min_v) * h).item())
        else:
            y0 = int(torch.floor(min_v * h).item())
            y1 = int(torch.ceil(max_v * h).item())
        x0, x1 = max(0, x0), min(w - 1, x1)
        y0, y1 = max(0, y0), min(h - 1, y1)
        if x1 < x0 or y1 < y0:
            continue

        x_candidates = xs[x0 : x1 + 1]
        y_candidates = ys[y0 : y1 + 1]
        gx, gy = torch.meshgrid(x_candidates, y_candidates, indexing="xy")
        uv_v = (1.0 - gy) if v_flip else gy
        pts = torch.stack([gx.reshape(-1), uv_v.reshape(-1)], dim=-1)

        v0 = p1 - p0
        v1 = p2 - p0
        v2 = pts - p0

        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = (v2 * v0).sum(dim=-1)
        d21 = (v2 * v1).sum(dim=-1)
        denom = d00 * d11 - d01 * d01
        if torch.abs(denom) < 1.0e-12:
            continue

        a = (d11 * d20 - d01 * d21) / denom
        b = (d00 * d21 - d01 * d20) / denom
        c = 1.0 - a - b
        inside = (a >= -1.0e-4) & (b >= -1.0e-4) & (c >= -1.0e-4)
        if not inside.any():
            continue

        idx_inside = torch.where(inside)[0]
        local_w = x1 - x0 + 1
        xi = idx_inside % local_w
        yi = idx_inside // local_w
        xx = x0 + xi
        yy = y0 + yi

        tri_idx[yy, xx] = fi
        bary[yy, xx, 0] = c[idx_inside]
        bary[yy, xx, 1] = a[idx_inside]
        bary[yy, xx, 2] = b[idx_inside]

    valid = (tri_idx >= 0).to(dtype)
    return {"tri_idx": tri_idx, "bary": bary, "uv_valid_mask": valid.unsqueeze(0).unsqueeze(0)}


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    normals = torch.zeros_like(vertices)
    tri = vertices[faces]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    for i in range(3):
        normals.index_add_(0, faces[:, i], fn)
    return F.normalize(normals, dim=-1, eps=1e-8)


def _interpolate_uv_attribute(vertex_attr: torch.Tensor, uv_mesh_faces: torch.Tensor, tri_idx: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    h, w = tri_idx.shape
    out = torch.zeros((h, w, vertex_attr.shape[-1]), device=vertex_attr.device, dtype=vertex_attr.dtype)
    valid = tri_idx >= 0
    if valid.any():
        fi = tri_idx[valid]
        face_vid = uv_mesh_faces[fi]
        vals = vertex_attr[face_vid]
        bw = bary[valid].unsqueeze(-1)
        out[valid] = (vals * bw).sum(dim=1)
    return out


def _project_points_world2cam(points: torch.Tensor, intrinsics: torch.Tensor, world2cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = points.shape[0]
    pts_h = torch.cat([points, torch.ones((n, 1), device=points.device, dtype=points.dtype)], dim=-1)
    cam = (world2cam @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    z_safe = torch.clamp(z, min=1.0e-6)
    uv_h = (intrinsics @ torch.stack([cam[:, 0] / z_safe, cam[:, 1] / z_safe, torch.ones_like(z_safe)], dim=-1).T).T
    return uv_h[:, :2], z


def _inside_mask(
    proj_xy: torch.Tensor,
    proj_z: torch.Tensor,
    valid_flat: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    return (
        (proj_xy[:, 0] >= 0)
        & (proj_xy[:, 0] <= (w - 1))
        & (proj_xy[:, 1] >= 0)
        & (proj_xy[:, 1] <= (h - 1))
        & (proj_z > 0)
        & valid_flat
    )


def _select_world2cam(
    vertices: torch.Tensor,
    intrinsics: torch.Tensor,
    transform_matrix: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    proj_xy_w2c, proj_z_w2c = _project_points_world2cam(vertices, intrinsics, transform_matrix)
    proj_xy_c2w, proj_z_c2w = _project_points_world2cam(vertices, intrinsics, torch.linalg.inv(transform_matrix))
    valid = torch.ones((vertices.shape[0],), dtype=torch.bool, device=vertices.device)
    inside_w2c = _inside_mask(proj_xy_w2c, proj_z_w2c, valid, h, w).sum()
    inside_c2w = _inside_mask(proj_xy_c2w, proj_z_c2w, valid, h, w).sum()
    return torch.linalg.inv(transform_matrix) if inside_c2w > inside_w2c else transform_matrix


def _rasterize_depth_image(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    world2cam: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    proj_xy, proj_z = _project_points_world2cam(vertices, intrinsics, world2cam)
    depth = torch.full((h, w), float("inf"), device=vertices.device, dtype=vertices.dtype)
    xs = torch.arange(w, device=vertices.device, dtype=vertices.dtype) + 0.5
    ys = torch.arange(h, device=vertices.device, dtype=vertices.dtype) + 0.5

    for fi in range(faces.shape[0]):
        tri_vid = faces[fi]
        tri_xy = proj_xy[tri_vid]
        tri_z = proj_z[tri_vid]
        if (tri_z <= 0).any():
            continue

        min_x = int(torch.floor(torch.min(tri_xy[:, 0])).item())
        max_x = int(torch.ceil(torch.max(tri_xy[:, 0])).item())
        min_y = int(torch.floor(torch.min(tri_xy[:, 1])).item())
        max_y = int(torch.ceil(torch.max(tri_xy[:, 1])).item())
        min_x, max_x = max(0, min_x), min(w - 1, max_x)
        min_y, max_y = max(0, min_y), min(h - 1, max_y)
        if max_x < min_x or max_y < min_y:
            continue

        gx, gy = torch.meshgrid(xs[min_x : max_x + 1], ys[min_y : max_y + 1], indexing="xy")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        p0, p1, p2 = tri_xy[0], tri_xy[1], tri_xy[2]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = pts - p0
        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = (v2 * v0).sum(dim=-1)
        d21 = (v2 * v1).sum(dim=-1)
        denom = d00 * d11 - d01 * d01
        if torch.abs(denom) < 1.0e-12:
            continue
        a = (d11 * d20 - d01 * d21) / denom
        b = (d00 * d21 - d01 * d20) / denom
        c = 1.0 - a - b
        inside = (a >= -1.0e-4) & (b >= -1.0e-4) & (c >= -1.0e-4)
        if not inside.any():
            continue

        idx_inside = torch.where(inside)[0]
        local_w = max_x - min_x + 1
        xi = idx_inside % local_w
        yi = idx_inside // local_w
        xx = min_x + xi
        yy = min_y + yi
        z_interp = c[idx_inside] * tri_z[0] + a[idx_inside] * tri_z[1] + b[idx_inside] * tri_z[2]
        current = depth[yy, xx]
        update = z_interp < current
        if update.any():
            depth[yy[update], xx[update]] = z_interp[update]

    return depth


def _sample_view(
    feat_view: torch.Tensor,
    intrinsics: torch.Tensor,
    world2cam: torch.Tensor,
    uv_points: torch.Tensor,
    uv_normals_flat: torch.Tensor,
    valid_flat: torch.Tensor,
    uv_resolution: int,
    use_front_facing: bool,
    depth_map: torch.Tensor | None = None,
    depth_abs_tol: float = 2.0e-3,
    depth_rel_tol: float = 1.0e-2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, h, w = feat_view.shape
    proj_xy, proj_z = _project_points_world2cam(uv_points, intrinsics, world2cam)

    gx = ((proj_xy[:, 0] + 0.5) / float(w)) * 2.0 - 1.0
    gy = ((proj_xy[:, 1] + 0.5) / float(h)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, uv_resolution, uv_resolution, 2)
    sampled = F.grid_sample(
        feat_view.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]

    inside = _inside_mask(proj_xy, proj_z, valid_flat, h, w)

    if use_front_facing:
        cam2world = torch.linalg.inv(world2cam)
        cam_origin = cam2world[:3, 3]
        view_dir = F.normalize(cam_origin.unsqueeze(0) - uv_points, dim=-1, eps=1e-8)
        front_facing = (uv_normals_flat * view_dir).sum(dim=-1) > 0
        inside = inside & front_facing

    if depth_map is not None:
        px = torch.round(proj_xy[:, 0]).to(torch.long).clamp(0, w - 1)
        py = torch.round(proj_xy[:, 1]).to(torch.long).clamp(0, h - 1)
        depth_ref = depth_map[py, px]
        depth_valid = torch.isfinite(depth_ref)
        depth_tol = torch.maximum(depth_ref * depth_rel_tol, torch.full_like(depth_ref, depth_abs_tol))
        depth_consistent = depth_valid & (torch.abs(proj_z - depth_ref) <= depth_tol)
        inside = inside & depth_consistent

    return sampled, inside.view(uv_resolution, uv_resolution).to(feat_view.dtype), proj_xy.view(uv_resolution, uv_resolution, 2)


def project_image_features_to_surface(
    image_features: torch.Tensor,
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    intrinsics: torch.Tensor,
    transform_matrices: torch.Tensor,
    template_mesh_path: str,
    uv_resolution: int = 256,
    uv_vertices: torch.Tensor | None = None,
    uv_faces: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if image_features.ndim != 5:
        raise ValueError(f"Expected image_features [B,V,C,H,W], got {tuple(image_features.shape)}")

    b, v, c, _, _ = image_features.shape
    device = image_features.device
    dtype = image_features.dtype

    if uv_vertices is None or uv_faces is None:
        uv_template = load_uv_template(template_mesh_path)
        uv_vertices = uv_template.uv_vertices.to(device=device, dtype=dtype)
        uv_faces = uv_template.uv_faces.to(device=device, dtype=torch.long)
        uv_mesh_faces = uv_template.mesh_faces_from_obj.to(device=device, dtype=torch.long)
    else:
        uv_vertices = uv_vertices.to(device=device, dtype=dtype)
        uv_faces = uv_faces.to(device=device, dtype=torch.long)
        uv_mesh_faces = uv_faces

    faces = mesh_faces.to(device=device, dtype=torch.long)

    # primary convention: no v-flip. A/B debug: also compute v-flip when single-view.
    rast = _rasterize_uv(uv_vertices, uv_faces, uv_resolution, v_flip=False)
    tri_idx = rast["tri_idx"]
    bary = rast["bary"]
    uv_valid_mask = rast["uv_valid_mask"].repeat(b, 1, 1, 1)

    uv_features = torch.zeros((b, v, c, uv_resolution, uv_resolution), device=device, dtype=dtype)
    uv_visibility = torch.zeros((b, v, 1, uv_resolution, uv_resolution), device=device, dtype=dtype)
    uv_position_map = torch.zeros((b, 3, uv_resolution, uv_resolution), device=device, dtype=dtype)
    uv_normal_map = torch.zeros((b, 3, uv_resolution, uv_resolution), device=device, dtype=dtype)

    # Debug outputs for single-view acceptance.
    uv_feature_single = torch.zeros((b, c, uv_resolution, uv_resolution), device=device, dtype=dtype)
    uv_visibility_single = torch.zeros((b, 1, uv_resolution, uv_resolution), device=device, dtype=dtype)
    uv_face_index = tri_idx.to(dtype).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
    uv_barycentric_vis = bary.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
    uv_feature_single_vflip = torch.zeros((b, c, uv_resolution, uv_resolution), device=device, dtype=dtype)
    uv_sample_xy = torch.zeros((b, 2, uv_resolution, uv_resolution), device=device, dtype=dtype)

    for bi in range(b):
        verts = mesh_vertices[bi]
        v_normals = _vertex_normals(verts, faces)

        uv_pos = _interpolate_uv_attribute(verts, uv_mesh_faces, tri_idx, bary)
        uv_nrm = F.normalize(_interpolate_uv_attribute(v_normals, uv_mesh_faces, tri_idx, bary), dim=-1, eps=1e-8)

        uv_position_map[bi] = uv_pos.permute(2, 0, 1)
        uv_normal_map[bi] = uv_nrm.permute(2, 0, 1)

        uv_points = uv_pos.view(-1, 3)
        uv_normals_flat = uv_nrm.view(-1, 3)
        valid_flat = uv_valid_mask[bi, 0].view(-1) > 0

        for vi in range(v):
            _, h_img, w_img = image_features[bi, vi].shape
            world2cam = _select_world2cam(
                vertices=verts,
                intrinsics=intrinsics[bi, vi],
                transform_matrix=transform_matrices[bi, vi],
                h=h_img,
                w=w_img,
            )
            depth_map = _rasterize_depth_image(
                vertices=verts,
                faces=faces,
                intrinsics=intrinsics[bi, vi],
                world2cam=world2cam,
                h=h_img,
                w=w_img,
            ) if v == 1 else None
            sampled, vis, sample_xy = _sample_view(
                feat_view=image_features[bi, vi],
                intrinsics=intrinsics[bi, vi],
                world2cam=world2cam,
                uv_points=uv_points,
                uv_normals_flat=uv_normals_flat,
                valid_flat=valid_flat,
                uv_resolution=uv_resolution,
                use_front_facing=True,
                depth_map=depth_map,
            )
            uv_features[bi, vi] = sampled
            uv_visibility[bi, vi, 0] = vis
            if vi == 0:
                uv_sample_xy[bi] = sample_xy.permute(2, 0, 1)

        # debug maps from first view
        uv_feature_single[bi] = uv_features[bi, 0]
        uv_visibility_single[bi, 0] = uv_visibility[bi, 0, 0]

        if v == 1:
            # minimal A/B test: flip UV-v and re-sample.
            rast_flip = _rasterize_uv(uv_vertices, uv_faces, uv_resolution, v_flip=True)
            uv_pos_flip = _interpolate_uv_attribute(verts, uv_mesh_faces, rast_flip["tri_idx"], rast_flip["bary"]).view(-1, 3)
            sampled_flip, _, _ = _sample_view(
                feat_view=image_features[bi, 0],
                intrinsics=intrinsics[bi, 0],
                world2cam=world2cam,
                uv_points=uv_pos_flip,
                uv_normals_flat=uv_normals_flat,
                valid_flat=(rast_flip["uv_valid_mask"][0, 0].view(-1) > 0),
                uv_resolution=uv_resolution,
                use_front_facing=True,
                depth_map=depth_map,
            )
            uv_feature_single_vflip[bi] = sampled_flip

    return {
        "uv_features": uv_features,
        "uv_visibility": uv_visibility,
        "uv_valid_mask": uv_valid_mask,
        "uv_position_map": uv_position_map,
        "uv_normal_map": uv_normal_map,
        "uv_feature_single": uv_feature_single,
        "uv_visibility_single": uv_visibility_single,
        "uv_face_index": uv_face_index,
        "uv_barycentric_vis": uv_barycentric_vis,
        "uv_feature_single_vflip": uv_feature_single_vflip,
        "uv_sample_xy": uv_sample_xy,
    }


def project_surface_to_uv(projection_out: Dict[str, torch.Tensor], confidence: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    uv_features = projection_out["uv_features"]
    uv_visibility = projection_out["uv_visibility"]
    if confidence is None:
        confidence = torch.ones_like(uv_visibility)

    return {
        "uv_features": uv_features,
        "visibility": uv_visibility,
        "confidence": confidence,
        "uv_valid_mask": projection_out["uv_valid_mask"],
        "uv_position_map": projection_out["uv_position_map"],
        "uv_normal_map": projection_out["uv_normal_map"],
        "uv_feature_single": projection_out["uv_feature_single"],
        "uv_visibility_single": projection_out["uv_visibility_single"],
        "uv_face_index": projection_out["uv_face_index"],
        "uv_barycentric_vis": projection_out["uv_barycentric_vis"],
        "uv_feature_single_vflip": projection_out["uv_feature_single_vflip"],
        "uv_sample_xy": projection_out["uv_sample_xy"],
    }
