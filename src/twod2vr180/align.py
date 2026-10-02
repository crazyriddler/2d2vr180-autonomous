"""Gravity alignment: find the scene's "up" so that viewers and VR180 renders
start level, at the eye position of the capture camera.

A photo is rarely taken perfectly level; reconstructions live in the camera
frame, so without this step the horizon is tilted and floors slope in VR.
Up is estimated from geometry: the scene is voxelised, each voxel's points
give a local plane (PCA); planar voxels whose normal is within 35° of the
camera's up vote for the gravity direction (floors, ceilings, table tops,
ground). With too little horizontal structure the camera's own up is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UpEstimate:
    up: np.ndarray            # unit vector, scene coordinates
    confidence: float         # fraction of planar area supporting it
    method: str               # "geometry" | "camera"
    tilt_deg: float           # angle between the camera's up and the estimated up

    def to_dict(self) -> dict:
        return {"up": [round(float(v), 6) for v in self.up], "confidence": round(self.confidence, 4),
                "method": self.method, "tilt_deg": round(self.tilt_deg, 2)}


def _ref_c2w(scene) -> np.ndarray:
    return np.asarray(scene.cameras[0].c2w if getattr(scene, "cameras", None) else np.eye(4), np.float64)


def _voxel_normals(points: np.ndarray, voxel: float, min_pts: int = 8) -> np.ndarray:
    """Normals of planar voxels (N,3)."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    nv = len(counts)
    s1 = np.zeros((nv, 3))
    s2 = np.zeros((nv, 3, 3))
    np.add.at(s1, inv, points)
    np.add.at(s2, inv, points[:, :, None] * points[:, None, :])
    ok = counts >= min_pts
    c = counts[ok][:, None]
    mean = s1[ok] / c
    cov = s2[ok] / c[:, :, None] - mean[:, :, None] * mean[:, None, :]
    w, v = np.linalg.eigh(cov)                      # ascending eigenvalues
    planar = w[:, 0] < 0.08 * np.maximum(w[:, 1], 1e-18)
    return v[planar, :, 0]


def estimate_up(scene, max_points: int = 400_000, seed: int = 0) -> UpEstimate:
    c2w = _ref_c2w(scene)
    prior = -c2w[:3, 1]                    # OpenCV: y points down in the camera
    prior = prior / np.linalg.norm(prior)
    means = np.asarray(scene.means, np.float64)
    ok = np.isfinite(means).all(1)
    if getattr(scene, "opacity", None) is not None:
        ok &= np.asarray(scene.opacity) > 0.2
    pts = means[ok]
    fallback = UpEstimate(prior, 0.0, "camera", 0.0)
    if len(pts) < 500:
        return fallback
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    depth = (pts - c2w[:3, 3]) @ c2w[:3, 2]
    dpos = depth[depth > 0]
    scale = float(np.median(dpos)) if len(dpos) else float(np.median(np.linalg.norm(pts - pts.mean(0), axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        return fallback
    normals = np.concatenate([_voxel_normals(pts, scale / v) for v in (40.0, 20.0)], 0)
    if len(normals) < 20:
        return fallback
    normals *= np.sign(normals @ prior)[:, None] + (normals @ prior == 0)[:, None]
    d = prior
    inl = np.zeros(len(normals), bool)
    for ang in (35.0, 20.0, 12.0, 8.0):
        inl = normals @ d > np.cos(np.radians(ang))
        if inl.sum() < 10:
            return fallback
        m = normals[inl].sum(0)
        d = m / np.linalg.norm(m)
    support = float(inl.mean())
    tilt = float(np.degrees(np.arccos(np.clip(d @ prior, -1, 1))))
    if support < 0.04 or tilt > 35:
        return UpEstimate(prior, support, "camera", 0.0)
    return UpEstimate(d, support, "geometry", tilt)


def level_c2w(c2w: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Same position and heading as ``c2w`` (OpenCV), with roll and pitch removed."""
    c2w = np.asarray(c2w, np.float64)
    u = np.asarray(up, np.float64) / np.linalg.norm(up)
    f = c2w[:3, 2] - (c2w[:3, 2] @ u) * u
    if np.linalg.norm(f) < 1e-3:              # looking straight up/down: use the image's up as heading
        f = -c2w[:3, 1] - (-c2w[:3, 1] @ u) * u
        if np.linalg.norm(f) < 1e-6:
            f = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    f /= np.linalg.norm(f)
    r = np.cross(f, u)      # OpenCV: right = down x forward = forward x up
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = r, -u, f, c2w[:3, 3]
    return out


def viewer_transform(scene, up: UpEstimate | None = None) -> dict:
    """Rigid transform scene → three.js world (y up, -z forward) that puts the
    capture camera at the origin, level, looking down -z. Returns a quaternion
    (x, y, z, w), a translation and the up estimate."""
    up = up or estimate_up(scene)
    L = level_c2w(_ref_c2w(scene), up.up)
    w2l = np.linalg.inv(L)                                  # scene → level OpenCV camera
    R = np.diag([1.0, -1.0, -1.0]) @ w2l[:3, :3]           # OpenCV camera → three.js
    t = np.diag([1.0, -1.0, -1.0]) @ w2l[:3, 3]
    return {"quaternion": rotmat_to_quat_xyzw(R), "position": [float(v) for v in t], "up": up.to_dict()}


def rotmat_to_quat_xyzw(R: np.ndarray) -> list[float]:
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return [float(v) for v in q]


def quat_xyzw_to_rotmat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
