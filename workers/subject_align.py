"""Place a generated 3D subject (TRELLIS.2 mesh, its own normalised frame) where the subject is in the photo.

TRELLIS.2 builds the whole subject - back included - from the photo, but in a canonical frame: unknown
orientation, a unit cube, no relation to the photo's camera. The photo's metric depth (MoGe-2, already in
the multi-view scene's units) and its subject mask fix that:

1. every candidate orientation (6 "up" axes x 24 turns) is tried; for each, scale and position follow from
   the mask (image size of the subject) and the depth (its distance);
2. the candidate whose silhouette and front surface best match the photo wins;
3. a scaled ICP between the mesh's front surface and the photo's subject points refines it.

The aligned mesh then gives depth of the whole subject - sides and back too - in any camera of the scene:
a geometry prior consistent all the way round, where per-view monocular depth is not.
Pure numpy (runs in any runtime; tested on the CPU)."""

from __future__ import annotations

import numpy as np


def sample_surface(vertices, faces, n, seed=0, colors=None):
    """``n`` points spread over the mesh surface by area, with their face normals (and, with per-vertex
    ``colors``, their interpolated colours as a third value)."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = a[idx] * (1 - s)[:, None] + b[idx] * (s * (1 - r2))[:, None] + c[idx] * (s * r2)[:, None]
    nrm = cr[idx] / np.maximum(area[idx], 1e-12)[:, None]
    if colors is not None:
        col = np.asarray(colors, np.float64)
        ca, cb, cc = col[f[idx, 0]], col[f[idx, 1]], col[f[idx, 2]]
        return pts, nrm, ca * (1 - s)[:, None] + cb * (s * (1 - r2))[:, None] + cc * (s * r2)[:, None]
    return pts, nrm


def render_depth(points, K, c2w, w, h, dilate=1):
    """Depth map (0 = empty) of a dense point set seen by a pinhole camera: nearest point per pixel,
    small holes between samples closed by a min-filter of radius ``dilate``."""
    c2w = np.asarray(c2w, np.float64)
    pc = (np.asarray(points, np.float64) - c2w[:3, 3]) @ c2w[:3, :3]
    z = pc[:, 2]
    ok = z > 1e-6
    u = np.floor(K[0][0] * pc[ok, 0] / z[ok] + K[0][2]).astype(np.int64)
    v = np.floor(K[1][1] * pc[ok, 1] / z[ok] + K[1][2]).astype(np.int64)
    z = z[ok]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.full(h * w, np.inf)
    np.minimum.at(d, v[inside] * w + u[inside], z[inside])
    d = d.reshape(h, w)
    if dilate:
        p = np.pad(d, dilate, constant_values=np.inf)
        best = d.copy()
        for dy in range(-dilate, dilate + 1):
            for dx in range(-dilate, dilate + 1):
                best = np.minimum(best, p[dilate + dy:dilate + dy + h, dilate + dx:dilate + dx + w])
        hole = ~np.isfinite(d)
        d[hole] = best[hole]
    d[~np.isfinite(d)] = 0.0
    return d


def candidate_rotations(turns=24):
    """Rotations taking the asset into the camera frame: each of the 6 axes as "up" (camera -y, OpenCV),
    and ``turns`` rotations about it."""
    ups = [np.array(a, float) for a in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    out = []
    for up in ups:
        # base rotation: asset "up" -> camera up (0, -1, 0)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(tmp, up)
        x /= np.linalg.norm(x)
        z = np.cross(x, up)
        B = np.stack([x, -up, z])          # rows: where asset x', up and z' go (camera x, -y, z)
        for k in range(turns):
            a = 2 * np.pi * k / turns
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            out.append(Ry @ B)
    return out


def _fit_scale_translation(pts_rot, mask, depth, K, iters=4):
    """Scale and translation that put rotated, centred points over the subject in the photo: the
    subject's median distance from the depth map, its image size and centre from the mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None
    z_med = float(np.median(depth[mask & (depth > 0)])) if np.any(mask & (depth > 0)) else None
    if not z_med:
        return None
    ext = pts_rot.max(0) - pts_rot.min(0)
    span_px = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    span_m = span_px * z_med / K[0][0]
    s = span_m / max(ext[0], ext[1], 1e-9)
    cu, cv = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    c = (pts_rot.max(0) + pts_rot.min(0)) / 2 * s
    depth_c = z_med + (c[2] - (pts_rot[:, 2].min() * s))  # front surface at the median distance
    t = np.array([(cu - K[0][2]) / K[0][0] * depth_c, (cv - K[1][2]) / K[1][1] * depth_c, depth_c]) - c
    # distance: move the subject along the camera rays (a scaling about the camera keeps its silhouette)
    # until its rendered front surface sits on the photo's depth
    # and size: grow or shrink about its own centre until the rendered silhouette spans the mask
    h, w = mask.shape
    span_mask = np.array([xs.max() - xs.min() + 1, ys.max() - ys.min() + 1], float)
    for _ in range(iters):
        d = render_depth(pts_rot * s + t, K, np.eye(4), w, h, dilate=1)
        both = (d > 0) & mask & (depth > 0)
        if both.sum() < 20:
            break
        ratio = float(np.median(depth[both] / d[both]))
        s, t = s * ratio, t * ratio
        d = render_depth(pts_rot * s + t, K, np.eye(4), w, h, dilate=1)
        ry, rx = np.nonzero(d > 0)
        if len(ry) < 20:
            break
        size = float(np.mean(span_mask / np.array([rx.max() - rx.min() + 1, ry.max() - ry.min() + 1], float)))
        s = s * size
        # and position in the image: the rendered silhouette centred on the mask
        du = cu - (rx.min() + rx.max() + 1) / 2
        dv = cv - (ry.min() + ry.max() + 1) / 2
        t = t + np.array([du / K[0][0] * t[2], dv / K[1][1] * t[2], 0.0])
        if abs(ratio - 1) < 2e-3 and abs(size - 1) < 2e-3 and abs(du) + abs(dv) < 0.5:
            break
    return s, t


def _score(pts_cam, mask, depth, K):
    h, w = mask.shape
    d = render_depth(pts_cam, K, np.eye(4), w, h, dilate=1)
    sil = d > 0
    inter = (sil & mask).sum()
    iou = inter / max((sil | mask).sum(), 1)
    both = sil & mask & (depth > 0)
    rel = float(np.median(np.abs(d[both] - depth[both]) / depth[both])) if both.sum() > 20 else 1.0
    return iou - 2.0 * rel


def umeyama_sim(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((a ** 2).sum() / len(src), 1e-12))
    return s, R, mu_d - s * R @ mu_s


def icp_refine(pts_cam, target, iters=20, trim=0.8):
    """Scaled ICP of the camera-facing part of ``pts_cam`` onto ``target`` (both camera frame).
    Returns the extra similarity (s, R, t) to apply after the current placement."""
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    s_tot, R_tot, t_tot = 1.0, np.eye(3), np.zeros(3)
    cur = pts_cam.copy()
    for _ in range(iters):
        dist, j = tree.query(cur)
        keep = dist <= np.quantile(dist, trim)
        s, R, t = umeyama_sim(cur[keep], target[j[keep]])
        cur = s * cur @ R.T + t
        s_tot, R_tot, t_tot = s * s_tot, R @ R_tot, s * R @ t_tot + t
    return s_tot, R_tot, t_tot


def front_points(pts_cam, K, w, h):
    """The points a camera at the origin sees (nearest per pixel)."""
    pc = pts_cam
    z = pc[:, 2]
    ok = z > 1e-6
    u = np.floor(K[0][0] * pc[:, 0] / np.where(ok, z, 1) + K[0][2]).astype(np.int64)
    v = np.floor(K[1][1] * pc[:, 1] / np.where(ok, z, 1) + K[1][2]).astype(np.int64)
    inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = render_depth(pc, K, np.eye(4), w, h, dilate=0)
    vis = inside.copy()
    vis[inside] = z[inside] <= d[v[inside], u[inside]] * 1.01 + 1e-9
    return vis


def align_to_photo(points, mask, depth, K, turns=24, work=160):
    """Similarity (s, R, t) mapping asset points into the photo's camera frame, and a score.
    ``mask``: subject in the photo (H, W bool); ``depth``: the photo's metric depth (H, W, 0 = unknown);
    ``K``: the photo's intrinsics. Candidates are scored on a ``work``-pixel-wide copy."""
    h, w = mask.shape
    f = work / w
    hs, ws = max(8, int(round(h * f))), work
    ys = np.minimum((np.arange(hs) + 0.5) / f, h - 1).astype(int)
    xs = np.minimum((np.arange(ws) + 0.5) / f, w - 1).astype(int)
    m_s, d_s = mask[ys][:, xs], depth[ys][:, xs]
    K_s = np.array(K, float).copy()
    K_s[:2] *= f
    P = np.asarray(points, np.float64)
    P = P - (P.max(0) + P.min(0)) / 2
    rng = np.random.default_rng(0)
    sub = P[rng.choice(len(P), min(len(P), 20000), replace=False)]
    best = None
    for R in candidate_rotations(turns):
        pr = sub @ R.T
        st = _fit_scale_translation(pr, m_s, d_s, K_s)
        if st is None:
            continue
        s, t = st
        sc = _score(pr * s + t, m_s, d_s, K_s)
        if best is None or sc > best[0]:
            best = (sc, s, R, t)
    if best is None:
        raise ValueError("no subject in the mask or no depth on it")
    sc, s, R, t = best
    # the chosen orientation: size and distance again at full resolution with all points
    st = _fit_scale_translation(P @ R.T, mask, depth, K, iters=12)
    if st is not None:
        s, t = st
    sub = P
    # refine against the photo's own subject points
    yy, xx = np.nonzero(mask & (depth > 0))
    pick = rng.choice(len(yy), min(len(yy), 30000), replace=False)
    yy, xx = yy[pick], xx[pick]
    z = depth[yy, xx]
    target = np.stack([(xx + 0.5 - K[0][2]) / K[0][0] * z, (yy + 0.5 - K[1][2]) / K[1][1] * z, z], 1)
    placed = s * sub @ R.T + t
    vis = front_points(placed, K, w, h)
    if vis.sum() > 100:
        s2, R2, t2 = icp_refine(placed[vis], target)
        s, R, t = s2 * s, R2 @ R, s2 * R2 @ t + t2
    centre = (np.asarray(points).max(0) + np.asarray(points).min(0)) / 2
    return {"scale": float(s), "R": R, "t": t - s * R @ centre, "score": float(sc)}


def apply(points, align):
    return align["scale"] * np.asarray(points, np.float64) @ align["R"].T + align["t"]


def place_subject(mesh_npz, mask, photo_depth, K0, cams, priors, generated, n_samples=200000, front_tol=0.9):
    """Align a TRELLIS.2 subject (``mesh_npz``) to the photo and use its depth as the prior of the AI views.

    ``mask``: the subject in the photo (at ``photo_depth``'s size); ``photo_depth``: the photo's depth in
    scene units (camera 0 = world frame); ``cams``: [(K, c2w, (w, h))] per view; ``priors``: per view
    depth prior or None, updated in place for generated views: inside the mesh's footprint its depth
    replaces the per-view monocular one, unless the monocular depth says something is clearly in front
    of the subject (kept). Returns (alignment, world points, colours) for the initial point cloud."""
    data = np.load(mesh_npz)
    pts, _, cols = sample_surface(data["vertices"], data["faces"], n_samples, colors=data["colors"])
    align = align_to_photo(pts, mask, photo_depth, K0)
    world = apply(pts, align)
    replaced = []
    for i, (K, c2w, (w, h)) in enumerate(cams):
        if not generated[i] or priors[i] is None:
            replaced.append(0.0)
            continue
        d = render_depth(world, K, c2w, w, h, dilate=2)
        p = priors[i]
        use = (d > 0) & ~((p > 0) & (p < d * front_tol))
        priors[i] = np.where(use, d, p).astype(np.float32)
        replaced.append(float(use.mean()))
    align["replaced"] = replaced
    return align, world.astype(np.float32), cols.astype(np.float32)
