"""Real 3D by SHARP fusion (rc55): one splat from the photo and 3 AI views, each given volume by Apple SHARP.

SHARP turns one image into 3D Gaussians in that image's own camera (OpenCV, its own metric guess). The
photo's splats are kept exactly as SHARP made them - they define the scene (owner: SHARP's front view is
the quality bar). Each AI view's splats are brought into the photo's frame and only what the photo cannot
show is added:

  1. pose: the camera engine's pose of the view (VGGT) - or the requested orbit around the subject when it
     is not installed - with the view's scale measured against the photo;
  2. refinement: a robust Sim(3) that lays the view's surfaces onto the photo's where both see the same
     thing (projective ICP: each point is pulled along the photo's line of sight onto the photo's surface);
  3. selection: a view splat is dropped when the photo (or an earlier view) already has a surface there,
     when it would sit in front of what the photo sees (it would hide the photo), or when it is one of
     SHARP's stretched "rubber sheet" splats across a depth edge of its own view. What is left lies behind
     the photo's surfaces or outside its frame: the sides of the subject and what it hides;
  4. colour: the view's exposure/white balance is matched to the photo on the shared surfaces.

Photo splats are INFERRED (single-image depth), added splats GENERATIVE. Pure numpy; runs in the app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .scene import GENERATIVE, INFERRED, Camera, GaussianScene, quat_to_rotmat

BUFFER_SIDE = 512          # depth-buffer resolution (long side) for visibility tests
MIN_OPACITY = 0.1          # splats fainter than this do not count as a surface


@dataclass
class FusionView:
    scene: GaussianScene                 # SHARP output in this image's camera frame
    camera: Camera                       # this image's intrinsics (SHARP metadata)
    generated: bool
    label: str = ""
    azimuth: float = 0.0                 # requested camera move (degrees; clockwise from above, + = above)
    elevation: float = 0.0
    pose: tuple | None = None            # initial (s, R, t) into the photo's frame, if known


@dataclass
class FusionResult:
    scene: GaussianScene
    poses: list = field(default_factory=list)       # (s, R, t) per view, photo first
    report: list = field(default_factory=list)


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


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Sim(3) (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def compose(a, b):
    """(a ∘ b): first b, then a."""
    sa, Ra, ta = a
    sb, Rb, tb = b
    return sa * sb, Ra @ Rb, sa * Ra @ tb + ta


def invert(p):
    s, R, t = p
    return 1 / s, R.T, -(R.T @ t) / s


def rotation_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


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

def scale_against(buf_depth: np.ndarray, ref_depth: np.ndarray) -> float | None:
    """Median ratio of two depth maps of the same view where both have a surface."""
    ok = np.isfinite(buf_depth) & np.isfinite(ref_depth) & (buf_depth > 0) & (ref_depth > 0)
    if ok.sum() < 200:
        return None
    return float(np.median(buf_depth[ok] / ref_depth[ok]))


def poses_from_engine(views: list[FusionView], c2ws, sizes, depths) -> list[tuple | None]:
    """Initial poses from the camera engine (VGGT): its cameras c2ws (photo = identity) and its depth maps
    of the images at sizes (w, h). Each SHARP view is scaled so its depth matches the
    engine's, and the engine's world is scaled to the photo's SHARP metres."""
    ratios = []
    for v, (w, h), d in zip(views, sizes, depths):
        # the same image at the engine's size: SHARP's lens, the engine's resolution
        sc = Camera(int(w), int(h), v.camera.fx * w / v.camera.width, v.camera.fy * h / v.camera.height,
                    v.camera.cx * w / v.camera.width, v.camera.cy * h / v.camera.height)
        buf = DepthBuffer(sc, max(w, h)).fill(v.scene.means, v.scene.opacity > MIN_OPACITY)
        ratios.append(scale_against(buf.depth, np.asarray(d, np.float64)))
    k = ratios[0]
    if not k:
        return [None] * len(views)
    out = []
    for i, (r, c2w) in enumerate(zip(ratios, c2ws)):
        c2w = np.asarray(c2w, np.float64)
        if i == 0:
            out.append((1.0, np.eye(3), np.zeros(3)))
        elif not r:
            out.append(None)
        else:
            out.append((k / r, c2w[:3, :3], k * c2w[:3, 3]))
    return out


def orbit_pose(azimuth: float, elevation: float, pivot_depth: float) -> tuple:
    """The camera the image model was asked for: an orbit around a pivot straight ahead of the photo's
    camera (azimuth about the vertical axis, + elevation = camera above, looking down)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(-e), -math.sin(-e)], [0, math.sin(-e), math.cos(-e)]])
    R = Ry @ Rx
    p = np.array([0.0, 0.0, pivot_depth])
    return 1.0, R, p - R @ p


class PhotoLevel:
    """The photo's visible surface at one resolution: 3D point and (normalised) colour per pixel."""

    def __init__(self, scene: GaussianScene, cam: Camera, side: int, colours: np.ndarray):
        self.buf = DepthBuffer(cam, side).fill(scene.means, scene.opacity > MIN_OPACITY)
        idx = self.buf.index
        ok = idx >= 0
        self.points = np.full((self.buf.h, self.buf.w, 3), np.nan, np.float32)
        self.points[ok] = scene.means[idx[ok]]
        self.colours = np.zeros((self.buf.h, self.buf.w, 3), np.float32)
        self.colours[ok] = colours[idx[ok]]


def normalised_colours(c: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Colours with each channel's mean and spread removed: exposure differences do not count."""
    ref = c if weight is None else c[weight]
    return (c - ref.mean(0)) / (ref.std(0) + 1e-3)


def photo_levels(scene: GaussianScene, cam: Camera, sides=(64, 128, 256, BUFFER_SIDE)) -> list[PhotoLevel]:
    cols = normalised_colours(scene.colors.astype(np.float64), scene.opacity > MIN_OPACITY)
    return [PhotoLevel(scene, cam, s, cols) for s in sides]


def _matches(P, C, lv: PhotoLevel, r: int, colour_sigma: float):
    """For every point, the photo surface point within r pixels that is closest in space and colour."""
    u, v, z, ok = lv.buf.project(P)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    yy = np.clip(v[None] + dy.reshape(-1, 1), 0, lv.buf.h - 1)
    xx = np.clip(u[None] + dx.reshape(-1, 1), 0, lv.buf.w - 1)
    Q = lv.points[yy, xx]                                   # (k, N, 3)
    px = 2.0 / lv.buf.fx                                    # two pixels, as a share of the depth
    dg = Q - P[None]
    g = np.einsum("knc,knc->kn", dg, dg) / (np.maximum(z, 1e-6) * px) ** 2
    dc = lv.colours[yy, xx] - C[None]
    cost = g + np.einsum("knc,knc->kn", dc, dc) / colour_sigma ** 2
    cost = np.where(np.isfinite(cost) & ok[None], cost, np.inf)
    k = cost.argmin(0)
    n = np.arange(len(P))
    return Q[k, n], cost[k, n], z


def refine_pose(pts: np.ndarray, cols: np.ndarray, init: tuple, levels: list[PhotoLevel], iters: int = 4,
                keep: float = 0.7, colour_sigma: float = 0.6):
    """Coloured ICP, coarse to fine: every point of the view is matched to the nearby photo surface point
    that is closest in space and colour, and a robust Sim(3) is fitted. Colour keeps the view from
    sliding along flat surfaces, which depth alone cannot see. Returns (pose, stats) with the share of
    points that end on the photo's surface."""
    cur = init
    for lv in levels:
        for _ in range(iters):
            s, R, t = cur
            P = s * pts @ R.T + t
            Q, cost, z = _matches(P, cols, lv, 3, colour_sigma)
            m = np.isfinite(cost)
            if m.sum() < 300:
                break
            idx = np.nonzero(m)[0]
            idx = idx[np.argsort(cost[idx])[: max(150, int(len(idx) * keep))]]
            step = umeyama(P[idx], Q[idx])
            cur = compose(step, cur)
    return cur, refine_stats(pts, cur, levels[-1].buf)


def refine_stats(pts: np.ndarray, pose: tuple, photo: DepthBuffer) -> dict:
    """How well a posed view lies on the photo's surface where it overlaps it."""
    s, R, t = pose
    u, v, z, ok = photo.project(s * pts @ R.T + t)
    d = photo.depth[v, u]
    ok &= np.isfinite(d)
    if not ok.sum():
        return {"matched": 0, "on_surface": 0.0, "median_error": None}
    rel = np.abs(z[ok] / d[ok] - 1)
    return {"matched": int(ok.sum()), "on_surface": float((rel < 0.05).mean()), "median_error": float(np.median(rel))}


def plausible_step(init: tuple, pose: tuple, max_turn: float, max_rescale: float) -> bool:
    r = pose[0] / init[0]
    return rotation_deg(pose[1] @ init[1].T) <= max_turn and 1 / max_rescale <= r <= max_rescale


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


def colour_match(src: np.ndarray, dst: np.ndarray):
    """Per-channel gain taking src colours to dst (robust, bounded; an offset is not fitted - on the few
    colours a surface has it trades off against the gain). Returns (gain, offset)."""
    gain, off = np.ones(3), np.zeros(3)
    if len(src) < 200:
        return gain, off
    for c in range(3):
        x, y = src[:, c].astype(np.float64), dst[:, c].astype(np.float64)
        g = 1.0
        for _ in range(3):
            g = float((x * y).sum() / max((x * x).sum(), 1e-9))
            r = np.abs(g * x - y)
            keep = r <= np.percentile(r, 70)
            x, y = x[keep], y[keep]
        gain[c] = float(np.clip(g, 0.75, 1.33))
    return gain, off


# ---------------------------------------------------------------- fusion

def fuse(views: list[FusionView], tol: float = 0.05, log=None, max_points: int = 20000,
         min_on_surface: float = 0.25, max_turn: float = 20.0, max_rescale: float = 1.5) -> FusionResult:
    """views[0] is the photo. Returns the photo's splats plus what each AI view adds."""
    log = log or (lambda m: None)
    photo = views[0]
    base = photo.scene.subset(np.ones(len(photo.scene), bool))
    base.provenance[:] = INFERRED
    base.sh_rest = None
    buf = DepthBuffer(photo.camera).fill(base.means, base.opacity > MIN_OPACITY)
    cells = occupied_cells(base.means[base.opacity > MIN_OPACITY], buf, tol)
    levels = photo_levels(base, photo.camera)
    centre = buf.depth[buf.h // 3: 2 * buf.h // 3, buf.w // 3: 2 * buf.w // 3]
    pivot = float(np.median(centre[np.isfinite(centre)])) if np.isfinite(centre).any() else 2.0
    parts, poses, report = [base], [(1.0, np.eye(3), np.zeros(3))], [{"view": photo.label or "photo",
                                                                       "splats": len(base), "kept": len(base)}]
    rng = np.random.default_rng(0)
    cams = [Camera(photo.camera.width, photo.camera.height, photo.camera.fx, photo.camera.fy, photo.camera.cx,
                   photo.camera.cy)]
    for v in views[1:]:
        sc = v.scene
        solid = sc.opacity > MIN_OPACITY
        idx = np.nonzero(solid)[0]
        if len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        pts = sc.means[idx].astype(np.float64)
        pcols = normalised_colours(sc.colors.astype(np.float64), solid)[idx]
        # initial pose: the camera engine's, else the requested orbit (both directions tried)
        inits = [("camera engine", v.pose)] if v.pose is not None else []
        if not inits:
            signs = (1, -1) if abs(v.azimuth) % 360 > 1e-3 else (1,)
            inits = [(f"requested orbit ({'+' if sg > 0 else '-'}azimuth)",
                      orbit_pose(sg * v.azimuth, v.elevation, pivot)) for sg in signs]
            # the view's own scale: its subject should sit at the pivot distance like the photo's
            vb = DepthBuffer(v.camera).fill(sc.means, solid)
            vc = vb.depth[vb.h // 3: 2 * vb.h // 3, vb.w // 3: 2 * vb.w // 3]
            if np.isfinite(vc).any():
                s0 = pivot / float(np.median(vc[np.isfinite(vc)]))
                inits = [(n, compose(p, (s0, np.eye(3), np.zeros(3)))) for n, p in inits]
        best = None
        for name, init in inits:
            pose, st = refine_pose(pts, pcols, init, levels)
            # a refinement that has to turn the camera a lot, or resize the view a lot, has slid into a
            # wrong fit (a mirrored orbit, a flat wall): not trusted
            if not plausible_step(init, pose, max_turn, max_rescale):
                if v.pose is not None:               # keep the camera engine's pose as it was
                    pose, st = init, refine_stats(pts, init, levels[-1].buf)
                    name += " (refinement rejected)"
                else:
                    continue
            if best is None or st["on_surface"] > best[2]["on_surface"]:
                best = (name, pose, st, init)
        if best is None:
            report.append({"view": v.label, "kept": 0,
                           "skipped": "its camera could not be found (no camera engine pose; the requested orbit "
                                      "does not fit the photo)"})
            log(f"{v.label}: skipped - its camera could not be found")
            poses.append(None)
            continue
        name, pose, st, init = best
        moved = rotation_deg(pose[1] @ init[1].T)
        rep = {"view": v.label, "pose_from": name, "scale": round(pose[0], 4),
               "refined_rotation_deg": round(moved, 2), "on_photo_surface": round(st["on_surface"], 3),
               "median_depth_error": None if st["median_error"] is None else round(st["median_error"], 4)}
        if st["on_surface"] < min_on_surface:
            rep.update(kept=0, skipped="it does not line up with the photo (too few shared surfaces)")
            report.append(rep)
            log(f"{v.label}: skipped - it does not line up with the photo ({st['on_surface']:.0%} on its surfaces)")
            poses.append(pose)
            continue
        poses.append(pose)
        world = transform_scene(sc, *pose)
        u, vv, z, inside = buf.project(world.means)
        d = buf.depth[vv, u]
        dup = near_occupied(world.means, buf, tol, cells)
        nb_min = neighbours(buf.depth).min(0)[vv, u]
        in_front = inside & np.isfinite(nb_min) & (z < nb_min * (1 - tol)) & ~dup
        sheet = rubber_sheet(sc, v.camera, tol=tol)
        faint = ~solid
        keep = ~dup & ~in_front & ~sheet & ~faint & (z > 1e-6)
        # colour: the view's splats on the photo's visible surface against the photo's own colours
        shared = inside & np.isfinite(d) & (np.abs(z / np.where(np.isfinite(d), d, 1) - 1) < tol) & solid
        pi = buf.index[vv[shared], u[shared]]
        okc = pi >= 0
        gain, off = colour_match(world.colors[shared][okc], base.colors[pi[okc]])
        add = world.subset(keep)
        add.colors = np.clip(add.colors * gain + off, 0, 1).astype(np.float32)
        add.provenance[:] = GENERATIVE
        parts.append(add)
        if len(add):
            cells = np.union1d(cells, occupied_cells(add.means, buf, tol))
        rep.update(splats=len(sc), kept=int(keep.sum()), duplicates=int((dup & solid).sum()),
                   in_front_of_photo=int(in_front.sum()), stretched=int((sheet & ~dup & solid).sum()),
                   colour_gain=[round(g, 3) for g in gain], colour_offset=[round(o, 3) for o in off])
        report.append(rep)
        log(f"{v.label}: pose from {name}, scale {pose[0]:.3f}, refined by {moved:.1f}°, "
            f"{st['on_surface']:.0%} on the photo's surfaces; adds {int(keep.sum()):,} of {len(sc):,} splats "
            f"({int((dup & solid).sum()):,} already in the photo, {int(in_front.sum()):,} in front of it, "
            f"{int((sheet & ~dup & solid).sum()):,} stretched)")
        s, R, t = pose
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, t
        cams.append(Camera(v.camera.width, v.camera.height, v.camera.fx, v.camera.fy, v.camera.cx, v.camera.cy,
                           c2w))
    scene = GaussianScene(np.concatenate([p.means for p in parts]), np.concatenate([p.scales for p in parts]),
                          np.concatenate([p.quats for p in parts]), np.concatenate([p.opacity for p in parts]),
                          np.concatenate([p.colors for p in parts]), np.concatenate([p.provenance for p in parts]),
                          None, cams, True)
    return FusionResult(scene, poses, report)


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
