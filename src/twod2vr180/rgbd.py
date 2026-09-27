"""Convert a per-pixel point map (monocular geometry backends) into
Gaussian splats and a textured OBJ mesh.

Every splat created here is tagged INFERRED: the colour was observed, the
depth was predicted by a network. Nothing is invented behind occluders, so
novel views will show holes there — which the VR180 renderer reports as
unknown coverage instead of hiding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .scene import INFERRED, Camera, GaussianScene


def _normals_from_points(p: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(p)
    dy = np.zeros_like(p)
    dx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    dy[1:-1, :] = p[2:, :] - p[:-2, :]
    dx[:, 0], dx[:, -1] = dx[:, 1], dx[:, -2]
    dy[0, :], dy[-1, :] = dy[1, :], dy[-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norm > 1e-12, n / np.maximum(norm, 1e-12), np.array([0, 0, -1.0]))
    # Face the camera (camera looks along +z, so visible normals have n.z < 0).
    flip = np.sum(n * p, axis=-1, keepdims=True) > 0
    return np.where(flip, -n, n)


def _quat_from_z_to(v: np.ndarray) -> np.ndarray:
    """Quaternions (w,x,y,z) rotating +z onto unit vectors v (N,3)."""
    z = np.array([0.0, 0.0, 1.0])
    d = v @ z
    axis = np.cross(np.broadcast_to(z, v.shape), v)
    w = 1.0 + d
    q = np.concatenate([w[:, None], axis], 1)
    anti = w < 1e-6  # v == -z: rotate 180deg about x
    q[anti] = np.array([0.0, 1.0, 0.0, 0.0])
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def depth_edges(depth: np.ndarray, rel_thresh: float = 0.05) -> np.ndarray:
    """Pixels on a depth discontinuity (relative jump to any 4-neighbour)."""
    z = depth
    e = np.zeros(z.shape, bool)
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                 ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        za, zb = z[a], z[b]
        jump = np.abs(za - zb) > rel_thresh * np.minimum(za, zb)
        e[a] |= jump
        e[b] |= jump
    return e


def pointmap_to_gaussians(points: np.ndarray, image: np.ndarray, mask: np.ndarray | None,
                          camera: Camera, footprint: float = 0.75, thickness: float = 0.15,
                          metric: bool = True, drop_edges: bool = True,
                          edge_rel: float = 0.04) -> GaussianScene:
    """points (H,W,3) camera-space, image (H,W,3) uint8, mask (H,W) bool.

    ``drop_edges`` removes "flying pixels" on depth discontinuities, which
    monocular networks place at interpolated (wrong) depths between the
    foreground and background."""
    h, w, _ = points.shape
    if image.shape[:2] != (h, w):
        image = np.asarray(Image.fromarray(image).resize((w, h), Image.LANCZOS))
    valid = np.isfinite(points).all(-1) & (points[..., 2] > 1e-6)
    if mask is not None:
        valid &= mask.astype(bool)
    if drop_edges:
        valid &= ~depth_edges(np.where(valid, points[..., 2], np.inf), edge_rel)
    normals = _normals_from_points(np.where(valid[..., None], points, 0.0))
    z = points[..., 2]
    fx = camera.fx * (w / camera.width)
    fy = camera.fy * (h / camera.height)
    s_xy = footprint * z / ((fx + fy) * 0.5)
    sel = valid.reshape(-1)
    means = points.reshape(-1, 3)[sel].astype(np.float32)
    s = s_xy.reshape(-1)[sel].astype(np.float32)
    # Grazing-angle surfels would become razor-thin; clamp the tilt used for
    # orientation so surfaces stay closed when seen from nearby viewpoints.
    n = normals.reshape(-1, 3)[sel]
    ray = means / np.linalg.norm(means, axis=1, keepdims=True)
    cosang = np.abs(np.sum(n * ray, 1))
    blend = np.clip((0.35 - cosang) / 0.35, 0, 1)[:, None]
    n = n * (1 - blend) + (-ray) * blend
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    scales = np.stack([s / np.maximum(cosang, 0.35) ** 0.5, s / np.maximum(cosang, 0.35) ** 0.5,
                       s * thickness], 1).astype(np.float32)
    quats = _quat_from_z_to(n.astype(np.float64))
    colors = (image.reshape(-1, 3)[sel].astype(np.float32) / 255.0)
    opacity = np.full(len(means), 0.98, np.float32)
    prov = np.full(len(means), INFERRED, np.uint8)
    cam = Camera(camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
    return GaussianScene(means, scales, quats, opacity, colors, prov, cameras=[cam], metric_scale=metric)


def pointmap_to_obj(points: np.ndarray, image: np.ndarray, mask: np.ndarray | None, out_obj: Path,
                    max_side: int = 768, edge_rel: float = 0.05) -> dict:
    """Write a textured grid mesh; triangles spanning depth discontinuities
    are dropped (no rubber-sheet surfaces across occlusion boundaries)."""
    h, w, _ = points.shape
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    p = points[::step, ::step]
    hh, ww, _ = p.shape
    valid = np.isfinite(p).all(-1) & (p[..., 2] > 1e-6)
    if mask is not None:
        valid &= mask[::step, ::step].astype(bool)
    edges = depth_edges(np.where(valid, p[..., 2], np.inf), edge_rel)
    idx = -np.ones((hh, ww), np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    ys, xs = np.nonzero(valid)
    verts = p[valid]
    uv = np.stack([(xs * step + 0.5) / w, 1.0 - (ys * step + 0.5) / h], 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    good = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    bad_edge = edges[:-1, :-1] & edges[1:, 1:] | edges[:-1, 1:] & edges[1:, :-1]
    zq = np.stack([p[:-1, :-1, 2], p[:-1, 1:, 2], p[1:, :-1, 2], p[1:, 1:, 2]], 0)
    with np.errstate(invalid="ignore"):
        spread = (np.nanmax(zq, 0) - np.nanmin(zq, 0)) / np.maximum(np.nanmin(zq, 0), 1e-9)
    good &= ~bad_edge & (spread < edge_rel * 2)
    t1 = np.stack([a[good], c[good], b[good]], 1)
    t2 = np.stack([b[good], c[good], d[good]], 1)
    faces = np.concatenate([t1, t2], 0) + 1
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_name = out_obj.stem + "_texture.png"
    Image.fromarray(image).save(out_obj.parent / tex_name)
    mtl_name = out_obj.stem + ".mtl"
    (out_obj.parent / mtl_name).write_text(
        f"newmtl source\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {tex_name}\n")
    # OBJ viewers expect y-up; flip OpenCV (y down, z forward) to y up, z back.
    v_out = verts * np.array([1.0, -1.0, -1.0])
    with open(out_obj, "w", newline="\n") as f:
        f.write(f"# 2D2VR180 mesh: geometry INFERRED from a monocular depth network\n")
        f.write(f"mtllib {mtl_name}\nusemtl source\n")
        np.savetxt(f, v_out, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, uv, fmt="vt %.5f %.5f")
        np.savetxt(f, np.repeat(faces, 2, axis=1)[:, [0, 1, 2, 3, 4, 5]], fmt="f %d/%d %d/%d %d/%d")
    return {"vertices": int(len(verts)), "faces": int(len(faces)), "texture": tex_name,
            "grid_step": step, "dropped_discontinuity_quads": int((~good).sum())}
