"""Geometry of Real 3D's fill (backends/locked.py): depth buffers of the photo's splat, what a moved view
sees that the photo never saw (unseen_by_photo), SHARP's stretched "rubber sheet" splats across depth
edges, and adding painted fills only where the photo has nothing (add_fills). Photo splats are INFERRED,
added splats GENERATIVE. Pure numpy; runs in the app.
"""

from __future__ import annotations

import math

import numpy as np

from .scene import GENERATIVE, Camera, GaussianScene

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


# ---------------------------------------------------------------- selection

def _cell_keys(pts: np.ndarray, buf: DepthBuffer, tol: float):
    """Cells of the photo's viewing frustum (pixel × log-depth step), extended beyond the frame."""
    z = np.maximum(pts[:, 2].astype(np.float64), 1e-6)
    u = np.floor(buf.fx * pts[:, 0] / z + buf.cx).astype(np.int64)
    v = np.floor(buf.fy * pts[:, 1] / z + buf.cy).astype(np.int64)
    k = np.floor(np.log(z) / math.log1p(tol)).astype(np.int64)
    return u, v, k


def _pack(u, v, k):
    off = 1 << 19
    return ((u + off) << 42) | ((v + off) << 21) | (k + (1 << 20))


def occupied_cells(pts: np.ndarray, buf: DepthBuffer, tol: float) -> np.ndarray:
    return np.unique(_pack(*_cell_keys(pts, buf, tol)))


def near_occupied(pts: np.ndarray, buf: DepthBuffer, tol: float, cells: np.ndarray) -> np.ndarray:
    """True where a point's cell or one of its 26 neighbours already holds a surface."""
    u, v, k = _cell_keys(pts, buf, tol)
    hit = np.zeros(len(pts), bool)
    if not len(cells):
        return hit
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            for dk in (-1, 0, 1):
                key = _pack(u + du, v + dv, k + dk)
                i = np.searchsorted(cells, key)
                hit |= cells[np.minimum(i, len(cells) - 1)] == key
    return hit


def rubber_sheet(scene: GaussianScene, cam: Camera, edge: float = 0.15, tol: float = 0.05) -> np.ndarray:
    """SHARP's splats stretched across a depth edge of their own view: at a pixel whose neighbourhood
    jumps in depth by more than ``edge``, splats strictly between the near and the far surface."""
    buf = DepthBuffer(cam).fill(scene.means, scene.opacity > MIN_OPACITY)
    nb = neighbours(buf.depth, 2)
    near = np.where(np.isfinite(nb), nb, np.inf).min(0)
    far = np.where(np.isfinite(nb), nb, -np.inf).max(0)
    u, v, z, ok = buf.project(scene.means)
    n0, f0 = near[v, u], far[v, u]
    jump = ok & np.isfinite(n0) & np.isfinite(f0) & (f0 > n0 * (1 + edge))
    return jump & (z > n0 * (1 + tol)) & (z < f0 * (1 - tol))


# ---------------------------------------------------------------- fusion


# ---------------------------------------------------------------- Real 3D v3: holes filled (rc57)

def unseen_by_photo(depth: np.ndarray, alpha: np.ndarray, cam: Camera, c2w: np.ndarray, photo: DepthBuffer,
                    tol: float = 0.04, min_alpha: float = 0.6) -> np.ndarray:
    """Pixels of a render (moved camera ``cam`` at ``c2w``, photo frame) that show nothing the photo saw:
    empty pixels, and surfaces behind what the photo sees (SHARP's guessed second layer, splats stretched
    across depth edges). These are the holes to fill."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.where(np.isfinite(depth), depth, 0.0).astype(np.float64)
    P = np.stack([(xs + 0.5 - cam.cx) / cam.fx * z, (ys + 0.5 - cam.cy) / cam.fy * z, z], -1).reshape(-1, 3)
    P = P @ np.asarray(c2w)[:3, :3].T + np.asarray(c2w)[:3, 3]
    u, v, z0, inside = photo.project(P)
    deepest = neighbours(photo.depth, pad=-np.inf).max(0)[v, u]
    behind = inside & np.isfinite(deepest) & (z0 > deepest * (1 + tol))
    empty = (alpha < min_alpha) | ~np.isfinite(depth) | (depth <= 0)
    return empty | behind.reshape(h, w)


def add_fills(base: GaussianScene, photo_cam: Camera, fills: list[tuple], tol: float = 0.05,
              log=None) -> tuple[GaussianScene, GaussianScene, list[dict]]:
    """Join the photo's splats with the splats made for the holes of moved views.

    ``fills``: (scene in the view's camera frame, view camera, c2w in the photo frame, hole mask, label).
    Only splats in a view's holes are taken; one is dropped when the photo or an earlier view already has a
    surface there, or when it would sit in front of what the photo sees. The photo's own guesses behind its
    surfaces (SHARP's second layer, stretched splats) are replaced where the new content lands.
    Returns (merged scene, added splats, report per view)."""
    log = log or (lambda m: None)
    solid = base.opacity > MIN_OPACITY
    buf = DepthBuffer(photo_cam).fill(base.means, solid)
    u, v, z, inside = buf.project(base.means)
    deepest = neighbours(buf.depth, pad=-np.inf).max(0)[v, u]
    hidden = inside & np.isfinite(deepest) & (z > deepest * (1 + tol))
    sheet = rubber_sheet(base, photo_cam, tol=tol)
    guessed = hidden | sheet
    cells = occupied_cells(base.means[solid & ~guessed], buf, tol)
    nb_min = neighbours(buf.depth).min(0)
    parts, report = [], []
    replaced = np.zeros(len(base), bool)
    gi = np.nonzero(guessed)[0]
    for sc, cam, c2w, mask, label in fills:
        z_v = sc.means[:, 2].astype(np.float64)
        ok = z_v > 1e-6
        zs = np.where(ok, z_v, 1.0)
        pu = np.floor(cam.fx * sc.means[:, 0] / zs + cam.cx).astype(np.int64)
        pv = np.floor(cam.fy * sc.means[:, 1] / zs + cam.cy).astype(np.int64)
        mh, mw = mask.shape
        ok &= (pu >= 0) & (pu < mw) & (pv >= 0) & (pv < mh) & (sc.opacity > MIN_OPACITY)
        in_hole = np.zeros(len(sc), bool)
        in_hole[ok] = mask[pv[ok], pu[ok]]
        c2w = np.asarray(c2w, np.float64)
        world = transform_scene(sc.subset(in_hole), 1.0, c2w[:3, :3], c2w[:3, 3])
        wu, wv, wz, win = buf.project(world.means)
        front = win & np.isfinite(nb_min[wv, wu]) & (wz < nb_min[wv, wu] * (1 - tol))
        dup = near_occupied(world.means, buf, tol, cells)
        keep = ~front & ~dup
        add = world.subset(keep)
        add.provenance[:] = GENERATIVE
        if len(add):
            parts.append(add)
            cells = np.union1d(cells, occupied_cells(add.means, buf, tol))
            # the photo's guesses this view drew in its holes give way to the new content
            w2c = np.linalg.inv(c2w)
            q = base.means[gi].astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
            qz = np.where(q[:, 2] > 1e-6, q[:, 2], 1.0)
            qu = np.floor(cam.fx * q[:, 0] / qz + cam.cx).astype(np.int64)
            qv = np.floor(cam.fy * q[:, 1] / qz + cam.cy).astype(np.int64)
            qok = (q[:, 2] > 1e-6) & (qu >= 0) & (qu < mw) & (qv >= 0) & (qv < mh)
            hit = np.zeros(len(gi), bool)
            hit[qok] = mask[qv[qok], qu[qok]]
            replaced[gi[hit]] = True
        rep = {"view": label, "in_holes": int(in_hole.sum()), "added": int(keep.sum()),
               "already_there": int(dup.sum()), "in_front_of_photo": int((front & ~dup).sum()),
               "hole_fraction": round(float(mask.mean()), 4)}
        report.append(rep)
        log(f"{label}: {rep['added']:,} new splats for its holes ({rep['hole_fraction']:.1%} of the view; "
            f"{rep['already_there']:,} already there, {rep['in_front_of_photo']:,} in front of the photo)")
    if parts:
        added = GaussianScene(*(np.concatenate([getattr(p, k) for p in parts])
                                for k in ("means", "scales", "quats", "opacity", "colors", "provenance")),
                              None, [], True)
        replaced |= guessed & near_occupied(base.means, buf, tol, occupied_cells(added.means, buf, tol))
    else:
        added = GaussianScene(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                              np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 3), np.float32),
                              np.zeros(0, np.uint8), None, [], True)
    kept = base.subset(~replaced)
    kept.sh_rest = None
    merged = GaussianScene(*(np.concatenate([getattr(kept, k), getattr(added, k)])
                             for k in ("means", "scales", "quats", "opacity", "colors", "provenance")),
                           None, list(base.cameras), True)
    if report:
        report.append({"view": "photo", "guessed_behind_surfaces": int(guessed.sum()),
                       "replaced_by_new_content": int(replaced.sum())})
        log(f"the photo's own guesses behind its surfaces: {int(guessed.sum()):,} splats, "
            f"{int(replaced.sum()):,} replaced by the new content")
    return merged, added, report
