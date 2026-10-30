"""Scene geometry helpers for Real 3D (backends/locked.py): depth buffers of a splat seen from a camera, camera
poses around a subject, and moving a splat in space. Pure numpy; runs in the app.

2.0.0-rc9: the AI fill of what a photo hides (unseen_by_photo, add_fills) was removed at the owner's request.
"""

from __future__ import annotations

import math

import numpy as np

from .scene import Camera, GaussianScene

BUFFER_SIDE = 512          # depth-buffer resolution (long side) for visibility tests
MIN_OPACITY = 0.1          # splats fainter than this do not count as a surface


# ---------------------------------------------------------------- geometry helpers

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z); a (4,) or (N,4), b (N,4)."""
    aw, ax, ay, az = np.moveaxis(np.broadcast_to(a, b.shape), -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def transform_scene(scene: GaussianScene, s: float, R: np.ndarray, t: np.ndarray) -> GaussianScene:
    """x -> s R x + t for every splat (sizes scale with s, orientations rotate with R)."""
    q = quat_mul(rotmat_to_quat(R), scene.quats.astype(np.float64))
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return GaussianScene((s * scene.means.astype(np.float64) @ R.T + t).astype(np.float32),
                         (scene.scales * s).astype(np.float32), q.astype(np.float32), scene.opacity.copy(),
                         scene.colors.copy(), scene.provenance.copy(), None, list(scene.cameras), scene.metric_scale)


class DepthBuffer:
    """Front-most splat per pixel of a camera (reduced resolution), for visibility tests."""

    def __init__(self, cam: Camera, side: int = BUFFER_SIDE):
        # never finer than the image: a splat per image pixel must cover the buffer without gaps
        f = min(side, max(cam.width, cam.height)) / max(cam.width, cam.height)
        self.w, self.h = max(8, round(cam.width * f)), max(8, round(cam.height * f))
        self.fx, self.fy = cam.fx * self.w / cam.width, cam.fy * self.h / cam.height
        self.cx, self.cy = cam.cx * self.w / cam.width, cam.cy * self.h / cam.height
        self.depth = np.full((self.h, self.w), np.inf)
        self.index = np.full((self.h, self.w), -1, np.int64)

    def project(self, pts: np.ndarray):
        """Pixel (u, v) as ints, depth z and whether the point is in front of the camera and inside."""
        z = pts[:, 2].astype(np.float64)
        ok = z > 1e-6
        zs = np.where(ok, z, 1.0)
        u = np.floor(self.fx * pts[:, 0] / zs + self.cx)
        v = np.floor(self.fy * pts[:, 1] / zs + self.cy)
        ok &= (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        return np.where(ok, u, 0).astype(np.int64), np.where(ok, v, 0).astype(np.int64), z, ok

    def fill(self, pts: np.ndarray, use: np.ndarray | None = None) -> "DepthBuffer":
        u, v, z, ok = self.project(pts)
        if use is not None:
            ok &= use
        idx = np.nonzero(ok)[0]
        flat = v[idx] * self.w + u[idx]
        order = np.lexsort((z[idx], flat))
        fs = flat[order]
        first = np.r_[True, fs[1:] != fs[:-1]]
        self.depth.flat[fs[first]] = z[idx][order][first]
        self.index.flat[fs[first]] = idx[order][first]
        self.depth = fill_holes(self.depth)
        return self


def neighbours(a: np.ndarray, r: int = 1, pad=np.inf) -> np.ndarray:
    """(k, h, w) stack of the (2r+1)^2 shifted copies of a."""
    p = np.pad(a, r, constant_values=pad)
    h, w = a.shape
    return np.stack([p[dy:dy + h, dx:dx + w] for dy in range(2 * r + 1) for dx in range(2 * r + 1)])


def fill_holes(d: np.ndarray, iters: int = 2) -> np.ndarray:
    """Empty pixels between splats take the nearest-surface depth of their 3x3 neighbourhood."""
    d = d.copy()
    for _ in range(iters):
        empty = ~np.isfinite(d)
        if not empty.any():
            break
        d[empty] = neighbours(d).min(0)[empty]
    return d


# ---------------------------------------------------------------- poses


def orbit_pose(azimuth: float, elevation: float, pivot_depth: float) -> tuple:
    """The camera the image model was asked for: an orbit around a pivot straight ahead of the photo's
    camera (azimuth about the vertical axis, + elevation = camera above, looking down)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(-e), -math.sin(-e)], [0, math.sin(-e), math.cos(-e)]])
    R = Ry @ Rx
    p = np.array([0.0, 0.0, pivot_depth])
    return 1.0, R, p - R @ p
