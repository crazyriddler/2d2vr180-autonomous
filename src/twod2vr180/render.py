"""CPU reference splat renderer (pinhole and VR180 half-equirectangular).

This renderer is dependency-free (numpy only) so that preview, VR180 stills
and acceptance checks work on any machine, including CI without a GPU. It
draws each Gaussian as an opaque isotropic splat with a z-buffer — an
approximation of true 3DGS alpha compositing that is honest about coverage:
pixels that no splat reaches stay *unknown* and are reported, never invented.
GPU-quality rendering (gsplat) lives inside backend runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import GaussianScene, quat_to_rotmat

MAX_RADIUS_PX = 12
SPLAT_K = 1.5  # disc radius in units of the projected sigma


@dataclass
class RenderResult:
    rgb: np.ndarray          # (H,W,3) uint8
    covered: np.ndarray      # (H,W) bool: some splat landed here
    filled: np.ndarray       # (H,W) bool: crack-filled by neighbour interpolation
    provenance: np.ndarray   # (H,W) uint8, 255 where uncovered
    depth: np.ndarray        # (H,W) float32, inf where uncovered
    clamped_splats: int = 0
    hole_filled: np.ndarray | None = None  # (H,W) bool: disocclusion filled from background

    def coverage_stats(self) -> dict:
        total = self.covered.size
        prov = self.provenance
        holes = self.hole_filled if self.hole_filled is not None else np.zeros_like(self.covered)
        out = {
            "covered_fraction": round(float(self.covered.mean()), 6),
            "crack_filled_fraction": round(float(self.filled.mean()), 6),
            "hole_filled_fraction": round(float(holes.mean()), 6),
            "unknown_fraction": round(float((~self.covered & ~self.filled & ~holes).mean()), 6),
        }
        for code, name in ((0, "observed"), (1, "inferred"), (2, "generative"), (3, "unknown_provenance")):
            out[f"{name}_fraction"] = round(float((prov == code).sum()) / total, 6)
        return out


def _perp_sigma(scene: GaussianScene, rays: np.ndarray) -> np.ndarray:
    """Isotropic sigma of each Gaussian projected on the plane normal to its
    viewing ray: sqrt(trace(P Σ P)/2) with P = I - r rᵀ."""
    R = quat_to_rotmat(scene.quats.astype(np.float64))
    s2 = scene.scales.astype(np.float64) ** 2
    # Σ = R diag(s2) Rᵀ ; trace(PΣP) = trace(Σ) - rᵀΣr
    tr = s2.sum(1)
    rR = np.einsum("ni,nij->nj", rays, R)
    rSr = np.sum(rR * rR * s2, 1)
    return np.sqrt(np.maximum(tr - rSr, 1e-18) / 2.0)


def _world_to_eye(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (points - t) @ R


def _rasterize(u, v, depth, radius_x, radius_y, ids, H, W):
    zbuf = np.full(H * W, np.inf, np.float64)
    rmax = int(min(MAX_RADIUS_PX, np.ceil(max(radius_x.max(initial=0), radius_y.max(initial=0)))))
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    fu = u - ui - 0.5
    fv = v - vi - 0.5
    offsets = [(dx, dy) for dy in range(-rmax, rmax + 1) for dx in range(-rmax, rmax + 1)]
    passes = []
    for dx, dy in offsets:
        ex = (dx - fu) / np.maximum(radius_x, 1e-6)
        ey = (dy - fv) / np.maximum(radius_y, 1e-6)
        inside = ex * ex + ey * ey <= 1.0
        if dx == 0 and dy == 0:
            inside[:] = True  # every splat covers at least its own pixel
        if not inside.any():
            continue
        x = ui[inside] + dx
        y = vi[inside] + dy
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not ok.any():
            continue
        pix = y[ok] * W + x[ok]
        d = depth[inside][ok]
        sid = ids[inside][ok]
        np.minimum.at(zbuf, pix, d)
        passes.append((pix, d, sid))
    idbuf = np.full(H * W, -1, np.int64)
    for pix, d, sid in passes:
        win = d <= zbuf[pix]
        idbuf[pix[win]] = sid[win]
    return zbuf, idbuf


def _crack_fill(rgb, covered, depth, iterations=1):
    filled = np.zeros_like(covered)
    for _ in range(iterations):
        H, W = covered.shape
        pad_c = np.pad(covered | filled, 1)
        cnt = sum(pad_c[1 + dy:1 + dy + H, 1 + dx:1 + dx + W].astype(np.int32)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
        target = ~(covered | filled) & (cnt >= 5)
        if not target.any():
            break
        pad_rgb = np.pad(rgb.astype(np.float32), ((1, 1), (1, 1), (0, 0)))
        acc = np.zeros(rgb.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                m = pad_c[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                acc += pad_rgb[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] * m[..., None]
        rgb[target] = (acc[target] / cnt[target][:, None]).round().clip(0, 255).astype(np.uint8)
        filled |= target
    return filled


def _finish(scene, zbuf, idbuf, H, W, background, crack_fill, clamped) -> RenderResult:
    covered = idbuf >= 0
    rgb = np.empty((H * W, 3), np.uint8)
    rgb[:] = np.asarray(background, np.uint8)
    rgb[covered] = (np.clip(scene.colors[idbuf[covered]], 0, 1) * 255 + 0.5).astype(np.uint8)
    prov = np.full(H * W, 255, np.uint8)
    prov[covered] = scene.provenance[idbuf[covered]]
    rgb = rgb.reshape(H, W, 3)
    covered = covered.reshape(H, W)
    depth = zbuf.reshape(H, W).astype(np.float32)
    filled = _crack_fill(rgb, covered, depth) if crack_fill else np.zeros_like(covered)
    return RenderResult(rgb, covered, filled, prov.reshape(H, W), depth, clamped)


def render_pinhole(scene: GaussianScene, c2w: np.ndarray, width: int, height: int, fx: float, fy: float,
                   cx: float, cy: float, near: float = 1e-3, background=(0, 0, 0),
                   crack_fill: bool = True, min_opacity: float = 0.05) -> RenderResult:
    p = _world_to_eye(scene.means.astype(np.float64), np.asarray(c2w, np.float64))
    z = p[:, 2]
    keep = (z > near) & (scene.opacity >= min_opacity)
    ids = np.nonzero(keep)[0]
    p = p[keep]
    z = z[keep]
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    rays_eye = p / np.linalg.norm(p, axis=1, keepdims=True)
    rays_world = rays_eye @ np.asarray(c2w)[:3, :3].T
    sig = _perp_sigma(scene.subset(keep), rays_world)
    r = SPLAT_K * sig * fx / z
    clamped = int((r > MAX_RADIUS_PX).sum())
    r = np.clip(r, 0.5, MAX_RADIUS_PX)
    margin = MAX_RADIUS_PX
    vis = (u > -margin) & (u < width + margin) & (v > -margin) & (v < height + margin)
    zbuf, idbuf = _rasterize(u[vis], v[vis], z[vis], r[vis], r[vis] * fy / fx, ids[vis], height, width)
    return _finish(scene, zbuf, idbuf, height, width, background, crack_fill, clamped)


def render_equirect180(scene: GaussianScene, c2w: np.ndarray, width: int, height: int,
                       near: float = 1e-3, background=(0, 0, 0), crack_fill: bool = True,
                       min_opacity: float = 0.05) -> RenderResult:
    """Half-equirectangular (VR180) eye image: longitude and latitude both
    span [-90°, 90°]; image centre looks along the camera's +z."""
    p = _world_to_eye(scene.means.astype(np.float64), np.asarray(c2w, np.float64))
    rng = np.linalg.norm(p, axis=1)
    keep = (rng > near) & (scene.opacity >= min_opacity) & (p[:, 2] > -1e-9)
    ids = np.nonzero(keep)[0]
    p = p[keep]
    rng = rng[keep]
    lon = np.arctan2(p[:, 0], p[:, 2])
    lat = np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2]))
    u = (lon / np.pi + 0.5) * width
    v = (0.5 - lat / np.pi) * height
    rays_world = (p / rng[:, None]) @ np.asarray(c2w)[:3, :3].T
    sig = _perp_sigma(scene.subset(keep), rays_world)
    ang = SPLAT_K * sig / rng
    ry = ang * height / np.pi
    rx = ang * width / np.pi / np.maximum(np.cos(lat), 0.05)
    clamped = int((np.maximum(rx, ry) > MAX_RADIUS_PX).sum())
    rx = np.clip(rx, 0.5, MAX_RADIUS_PX)
    ry = np.clip(ry, 0.5, MAX_RADIUS_PX)
    zbuf, idbuf = _rasterize(u, v, rng, rx, ry, ids, height, width)
    return _finish(scene, zbuf, idbuf, height, width, background, crack_fill, clamped)


def look_at_c2w(eye: np.ndarray, target: np.ndarray, down: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """OpenCV-convention camera-to-world looking from eye to target."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(down, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def orbit_c2w(center: np.ndarray, distance: float, yaw_deg: float, pitch_deg: float,
              base_c2w: np.ndarray | None = None) -> np.ndarray:
    """Camera orbiting ``center`` at ``distance``; yaw/pitch are relative to
    the reference camera (yaw 0, pitch 0 = reference viewing direction)."""
    base = np.eye(4) if base_c2w is None else np.asarray(base_c2w)
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    d_cam = np.array([np.sin(yaw) * np.cos(pitch), -np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    d = base[:3, :3] @ d_cam
    eye = center - d * distance
    return look_at_c2w(eye, center, down=base[:3, 1])


def fill_disocclusions(r: RenderResult, max_gap_frac: float = 0.06) -> RenderResult:
    """Fill horizontal gaps between rendered content with the colour of the
    farther (background) side — classic depth-image-based-rendering hole
    filling. Only gaps bounded on both sides are filled, so the region outside
    the reconstructed field of view stays unknown. Filled pixels are
    interpolation, not observation, and are reported as such."""
    H, W = r.covered.shape
    known = r.covered | r.filled
    cols = np.arange(W)[None, :].repeat(H, 0)
    left = np.maximum.accumulate(np.where(known, cols, -1), axis=1)
    right = np.where(known, cols, W)[:, ::-1]
    right = np.minimum.accumulate(right, axis=1)[:, ::-1]
    gap = (~known) & (left >= 0) & (right < W) & ((right - left - 1) <= max(1, int(max_gap_frac * W)))
    if not gap.any():
        r.hole_filled = np.zeros_like(known)
        return r
    ys, xs = np.nonzero(gap)
    lx, rx = left[ys, xs], right[ys, xs]
    # depth of crack-filled pixels is inf; use neighbouring rendered depth where needed
    dl = r.depth[ys, lx]
    dr = r.depth[ys, rx]
    dl = np.where(np.isfinite(dl), dl, -np.inf)
    dr = np.where(np.isfinite(dr), dr, -np.inf)
    src_x = np.where(dl >= dr, lx, rx)
    r.rgb[ys, xs] = r.rgb[ys, src_x]
    r.depth[ys, xs] = r.depth[ys, src_x]
    r.provenance[ys, xs] = r.provenance[ys, src_x]
    r.hole_filled = gap
    return r


# ----------------------------------------------------------------- cube maps
# Faces of a hemisphere cube map in the eye's OpenCV frame: (camera z, camera y).
CUBE_FACES = {
    "front": ((0, 0, 1), (0, 1, 0)),
    "right": ((1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, 1, 0)),
    "up": ((0, -1, 0), (0, 0, 1)),
    "down": ((0, 1, 0), (0, 0, -1)),
}


def face_rotation(name: str) -> np.ndarray:
    z, y = (np.asarray(v, np.float64) for v in CUBE_FACES[name])
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)  # columns = camera axes in eye coordinates


def cube_face_views(eye_c2w: np.ndarray, face_res: int) -> list[dict]:
    """Five 90° pinhole views covering the forward hemisphere of an eye."""
    views = []
    f = face_res / 2.0  # tan(45°) = 1
    for name in CUBE_FACES:
        c2w = np.eye(4)
        c2w[:3, :3] = eye_c2w[:3, :3] @ face_rotation(name)
        c2w[:3, 3] = eye_c2w[:3, 3]
        views.append({"face": name, "c2w": c2w, "width": face_res, "height": face_res,
                      "fx": f, "fy": f, "cx": face_res / 2.0, "cy": face_res / 2.0})
    return views


_LUT_CACHE: dict = {}


def _equirect_lut(out_res: int, face_res: int):
    key = (out_res, face_res)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    j, i = np.mgrid[0:out_res, 0:out_res]
    lon = ((i + 0.5) / out_res - 0.5) * np.pi
    lat = (0.5 - (j + 0.5) / out_res) * np.pi
    d = np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    best = np.full(d.shape[:2], -np.inf)
    face_idx = np.zeros(d.shape[:2], np.int64)
    u = np.zeros(d.shape[:2])
    v = np.zeros(d.shape[:2])
    for k, name in enumerate(CUBE_FACES):
        c = d @ face_rotation(name)  # eye → camera coordinates
        take = c[..., 2] > best
        best = np.where(take, c[..., 2], best)
        face_idx = np.where(take, k, face_idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(take, (c[..., 0] / c[..., 2] + 1) * 0.5 * face_res - 0.5, u)
            v = np.where(take, (c[..., 1] / c[..., 2] + 1) * 0.5 * face_res - 0.5, v)
    u = np.clip(u, 0, face_res - 1)
    v = np.clip(v, 0, face_res - 1)
    _LUT_CACHE[key] = (face_idx, u, v)
    return _LUT_CACHE[key]


def cubemap_to_equirect180(faces_rgb: np.ndarray, faces_alpha: np.ndarray, faces_depth: np.ndarray,
                           out_res: int, alpha_thresh: float = 0.5) -> RenderResult:
    """faces_* stacked in CUBE_FACES order: rgb (5,F,F,3) uint8, alpha (5,F,F) float [0,1],
    depth (5,F,F) float (range or z; used only for hole filling order)."""
    F = faces_rgb.shape[1]
    fi, u, v = _equirect_lut(out_res, F)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, F - 1)
    v1 = np.minimum(v0 + 1, F - 1)
    wu = (u - u0)[..., None]
    wv = (v - v0)[..., None]
    rgb = faces_rgb.astype(np.float32)
    c = (rgb[fi, v0, u0] * (1 - wu) * (1 - wv) + rgb[fi, v0, u1] * wu * (1 - wv)
         + rgb[fi, v1, u0] * (1 - wu) * wv + rgb[fi, v1, u1] * wu * wv)
    a = faces_alpha[fi, np.rint(v).astype(np.int64), np.rint(u).astype(np.int64)]
    dep = faces_depth[fi, np.rint(v).astype(np.int64), np.rint(u).astype(np.int64)].astype(np.float32)
    covered = a >= alpha_thresh
    out = np.zeros((out_res, out_res, 3), np.uint8)
    # un-premultiply partially transparent edges so they do not darken
    col = c / np.maximum(a[..., None], 1e-3)
    out[covered] = np.clip(col[covered], 0, 255).astype(np.uint8)
    depth = np.where(covered, dep, np.inf).astype(np.float32)
    prov = np.where(covered, 3, 255).astype(np.uint8)  # renderer cannot see per-splat provenance
    return RenderResult(out, covered, np.zeros_like(covered), prov, depth)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def interpolate_poses(c2ws: list[np.ndarray], n_out: int) -> list[np.ndarray]:
    """Resample a camera path to n_out poses (linear positions, slerp rotations)."""
    c2ws = [np.asarray(c, np.float64) for c in c2ws]
    if len(c2ws) == 1 or n_out <= 1:
        return [c2ws[0].copy() for _ in range(max(n_out, 1))]
    quats = [_rot_to_quat(c[:3, :3]) for c in c2ws]
    out = []
    for k in range(n_out):
        t = k * (len(c2ws) - 1) / (n_out - 1)
        i = min(int(np.floor(t)), len(c2ws) - 2)
        f = t - i
        q = _slerp(quats[i], quats[i + 1], f)
        m = np.eye(4)
        m[:3, :3] = quat_to_rotmat(q[None].astype(np.float64))[0]
        m[:3, 3] = (1 - f) * c2ws[i][:3, 3] + f * c2ws[i + 1][:3, 3]
        out.append(m)
    return out
