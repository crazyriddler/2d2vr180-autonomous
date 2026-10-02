"""Sharp multi-view fusion of per-view monocular geometry (numpy only).

Every view (the user's photo plus a few generated views) has a crisp, pixel-aligned depth map
(MoGe-2) that has been scaled into a shared, posed frame (VGGT). Instead of optimising one splat
model against all views - which averages the small inconsistencies of generated images into blur -
each pixel becomes a surfel exactly as in the single-photo backend, and views are merged by
*coverage*: the photo is kept whole, and a generated view only contributes surfaces that no
earlier view already shows (sides, top, underside, disoccluded background). Seen from the photo's
viewpoint the result is the photo; seen from elsewhere it is the nearest view that saw that surface.

Coordinates: OpenCV (x right, y down, z forward), scene frame = reference camera.
"""

import numpy as np

SH_C0 = 0.28209479177387814


def depth_edges(z, rel=0.04):
    """Pixels on a depth discontinuity (relative jump to a 4-neighbour)."""
    e = np.zeros(z.shape, bool)
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                 ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        za, zb = z[a], z[b]
        with np.errstate(invalid="ignore"):
            jump = np.abs(za - zb) > rel * np.minimum(za, zb)
        e[a] |= jump
        e[b] |= jump
    return e


def normals(p):
    """Camera-facing normals of a camera-space point map (H, W, 3)."""
    dx = np.zeros_like(p)
    dy = np.zeros_like(p)
    dx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    dy[1:-1, :] = p[2:, :] - p[:-2, :]
    dx[:, 0], dx[:, -1] = dx[:, 1], dx[:, -2]
    dy[0, :], dy[-1, :] = dy[1, :], dy[-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norm > 1e-12, n / np.maximum(norm, 1e-12), np.array([0, 0, -1.0], np.float32))
    flip = np.sum(n * p, axis=-1, keepdims=True) > 0
    return np.where(flip, -n, n)


def quat_from_z_to(v):
    """Quaternions (w, x, y, z) rotating +z onto unit vectors v (N, 3)."""
    d = v[:, 2]
    axis = np.stack([-v[:, 1], v[:, 0], np.zeros_like(d)], 1)   # z x v
    w = 1.0 + d
    q = np.concatenate([w[:, None], axis], 1)
    q[w < 1e-6] = np.array([0.0, 1.0, 0.0, 0.0])
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def camera_points(z, K):
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)


def resample(src, shape):
    """Nearest-neighbour resize of a 2-D map (no new depth values at edges)."""
    h, w = shape
    sh, sw = src.shape[:2]
    yi = np.minimum(((np.arange(h) + 0.5) * sh / h).astype(int), sh - 1)
    xi = np.minimum(((np.arange(w) + 0.5) * sw / w).astype(int), sw - 1)
    return src[yi][:, xi]


def align_scale(z_mono, z_ref, ok):
    """Robust scale s with s * z_mono ~ z_ref (median of ratios over trusted pixels)."""
    m = ok & np.isfinite(z_mono) & (z_mono > 0) & np.isfinite(z_ref) & (z_ref > 0)
    if m.sum() < 200:
        return None
    r = z_ref[m] / z_mono[m]
    lo, hi = np.percentile(r, [10, 90])
    r = r[(r >= lo) & (r <= hi)]
    return float(np.median(r)) if len(r) else None


class View:
    """One posed view: aligned depth z (H, W), intrinsics K, camera-to-world c2w, image (H, W, 3) uint8."""

    def __init__(self, z, K, c2w, image, valid, generated, grazing=0.3, edge_rel=0.04):
        self.z = np.where(valid, z, np.nan).astype(np.float32)
        self.K = np.asarray(K, np.float64)
        self.c2w = np.asarray(c2w, np.float64)
        self.w2c = np.linalg.inv(self.c2w)
        self.image = image
        self.generated = bool(generated)
        self.h, self.w = z.shape
        pc = camera_points(np.nan_to_num(self.z, nan=0.0), self.K)
        self.valid = valid & ~depth_edges(np.where(valid, self.z, np.inf), edge_rel)
        n = normals(np.where(self.valid[..., None], pc, 0.0))
        ray = pc / np.maximum(np.linalg.norm(pc, axis=-1, keepdims=True), 1e-9)
        self.cos = np.abs(np.sum(n * ray, -1))
        self.strong = self.valid & (self.cos >= grazing)
        self.pc, self.n = pc, n
        self.keep = np.zeros((self.h, self.w), bool)

    def world(self, sel):
        R, t = self.c2w[:3, :3], self.c2w[:3, 3]
        return (self.pc[sel] @ R.T + t).astype(np.float32)

    def project(self, X):
        """World points → (u, v) integer pixels, depth, in-bounds mask."""
        R, t = self.w2c[:3, :3], self.w2c[:3, 3]
        pc = X @ R.T + t
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(self.K[0, 0] * pc[:, 0] / z + self.K[0, 2])
            v = np.floor(self.K[1, 1] * pc[:, 1] / z + self.K[1, 2])
        inb = (z > 1e-6) & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        u = np.where(inb, u, 0).astype(np.int64)
        v = np.where(inb, v, 0).astype(np.int64)
        return u, v, z, inb


def fusion_order(c2ws, ref=0):
    """Reference first, then views by increasing viewpoint change (rotation angle + baseline)."""
    c0 = np.asarray(c2ws[ref])
    scale = max(1e-6, float(np.median([np.linalg.norm(np.asarray(m)[:3, 3] - c0[:3, 3]) for m in c2ws])))
    cost = []
    for i, m in enumerate(c2ws):
        m = np.asarray(m)
        cosang = np.clip((np.trace(c0[:3, :3].T @ m[:3, :3]) - 1) / 2, -1, 1)
        cost.append(np.degrees(np.arccos(cosang)) + 20 * np.linalg.norm(m[:3, 3] - c0[:3, 3]) / scale)
    cost[ref] = -1
    return [int(i) for i in np.argsort(cost, kind="stable")]


def candidates_ok(view, sel, accepted, tau_front=0.08, tau_back=0.10):
    """For the selected pixels of `view`, True where no accepted view already shows that surface.

    Projected into an accepted view, a point within [-tau_front, +tau_back] (relative) of the depth
    that view kept there is the same surface - small depth disagreements between views are normal -
    and is dropped. A point further in front would float in front of something that view saw: an
    inconsistency, also dropped. Only points well behind (disocclusions) or outside every accepted
    view are new."""
    X = view.world(sel)
    ok = np.ones(len(X), bool)
    for a in accepted:
        u, v, z, inb = a.project(X)
        za = a.z[v, u]
        has = inb & np.isfinite(za)
        with np.errstate(invalid="ignore"):
            rel = (z - za) / za
            covered = has & a.keep[v, u] & (rel >= -tau_front) & (rel <= tau_back)
            in_front = has & a.valid[v, u] & (rel < -tau_front)
        ok &= ~(covered | in_front)
    return ok


def fuse(views, order, tau_front=0.08, tau_back=0.10):
    """Mark in each view's `keep` the pixels that become splats. The first view in `order`
    (the photo) is kept whole; the others add what is not yet covered: first well-observed
    surfaces of every view, then grazing-angle ones."""
    accepted = []
    first = views[order[0]]
    first.keep[:] = first.valid
    accepted.append(first)
    for pass_strong in (True, False):
        for i in order[1:]:
            v = views[i]
            cand = (v.strong if pass_strong else (v.valid & ~v.strong)) & ~v.keep
            if not cand.any():
                if pass_strong:
                    accepted.append(v)
                continue
            ok = candidates_ok(v, cand, [a for a in accepted if a is not v], tau_front, tau_back)
            ys, xs = np.nonzero(cand)
            v.keep[ys[ok], xs[ok]] = True
            if pass_strong:
                accepted.append(v)
    return views


def surfels(view, footprint=0.75, thickness=0.15, stride_scale=1.0):
    """Gaussians for the kept pixels of a view: (means, log-scales, quats wxyz, logit-opacity, rgb)."""
    sel = view.keep
    means = view.world(sel)
    z = view.pc[..., 2][sel]
    f = 0.5 * (view.K[0, 0] + view.K[1, 1])
    s = footprint * stride_scale * z / f
    cos = np.maximum(view.cos[sel], 0.35)
    R = view.c2w[:3, :3]
    nc = view.n[sel]
    ray = view.pc[sel] / np.maximum(np.linalg.norm(view.pc[sel], axis=1, keepdims=True), 1e-9)
    blend = np.clip((0.35 - view.cos[sel]) / 0.35, 0, 1)[:, None]
    nc = nc * (1 - blend) + (-ray) * blend          # grazing surfels turn towards their camera
    nw = nc @ R.T
    nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-9)
    scales = np.stack([s / cos ** 0.5, s / cos ** 0.5, s * thickness], 1)
    rgb = view.image[sel].astype(np.float32) / 255.0
    return {"means": means, "scales": np.log(np.maximum(scales, 1e-8)).astype(np.float32),
            "quats": quat_from_z_to(nw.astype(np.float64)),
            "opacities": np.full(len(means), np.log(0.98 / 0.02), np.float32), "rgb": rgb}


def assemble(views, max_gaussians=None, seed=0):
    """Concatenate the surfels of all views; provenance 1 (inferred: photo pixels) or
    2 (generative: generated views). Over the cap, generated surfels are thinned (and enlarged)."""
    parts, prov = [], []
    for v in views:
        if v.keep.any():
            parts.append(surfels(v))
            prov.append(np.full(int(v.keep.sum()), 2 if v.generated else 1, np.uint8))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    prov = np.concatenate(prov)
    if max_gaussians and len(prov) > max_gaussians:
        gen = np.nonzero(prov == 2)[0]
        n_fixed = len(prov) - len(gen)
        room = max(0, max_gaussians - n_fixed)
        if len(gen) > room:
            rng = np.random.default_rng(seed)
            keep_gen = rng.choice(gen, room, replace=False) if room else np.zeros(0, np.int64)
            sel = np.concatenate([np.nonzero(prov != 2)[0], np.sort(keep_gen)])
            grow = np.log(np.sqrt(len(gen) / max(room, 1)))
            out = {k: a[sel] for k, a in out.items()}
            prov = prov[sel]
            out["scales"][prov == 2, :2] += grow
    out["sh0"] = ((out.pop("rgb") - 0.5) / SH_C0)[:, None, :].astype(np.float32)
    return out, prov
