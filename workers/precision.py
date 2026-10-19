"""Precision for Real 3D photos (2.0.0-rc7): truer depth, finer relief, clean edges.

Owner request: more precise 3D and depth from a photo. What each step does about a limit of the photo path:
  1. One pass of one depth model has its own errors and noise.
     -> MoGe-3 several times (the photo, its mirror image, a coarser pass), median in log depth
        (combine_passes); then a second, independent model (Apple Depth Pro) joined to it (depth_ensemble):
        the overall shape from both, the edges from Depth Pro, which is the sharpest at object boundaries.
  2. A wrong lens flattens or stretches the whole scene.
     -> the EXIF lens is checked against the two models' own estimates (lens_choice).
  3. Depth models smooth the fine relief: folds, fingers, the relief of a face.
     -> surface normals (MoGe-3's own; Sapiens2's for people) show that relief; they are joined to the depth
        as its slopes (integrate_normals: screened Poisson), the depth keeping the overall shape and scale.
  4. Monocular depth blurs depth edges, so splats land between a person and the background ("flying"
     splats: the trails seen in VR when moving the head).
     -> each pixel near a depth edge goes to the near or the far side, by its colour (resolve_edges).
numpy + torch only (runs in the 'photo-cu128' runtime, inside moge_worker and sharp_depth_worker).
"""

import numpy as np


# ---------------------------------------------------------------- helpers

def _t(a, torch, device=None):
    return torch.as_tensor(np.asarray(a, np.float32), device=device)


def log_depth(depth):
    """log of an (H, W) depth map, NaN where unknown or not positive."""
    d = np.asarray(depth, np.float64)
    ok = np.isfinite(d) & (d > 0)
    return np.where(ok, np.log(np.where(ok, d, 1.0)), np.nan)


def rays(h, w, fx, fy=None, cx=None, cy=None):
    """Per-pixel ray directions (H, W, 3) with z = 1 for a pinhole camera (OpenCV axes)."""
    fy = fx if fy is None else fy
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    u = (np.arange(w) + 0.5 - cx) / fx
    v = (np.arange(h) + 0.5 - cy) / fy
    return np.stack(np.broadcast_arrays(u[None, :], v[:, None], np.ones((h, w))), -1)


def blur(a, sigma, torch):
    """NaN-aware Gaussian blur (detail_geometry.blur_nan)."""
    from detail_geometry import blur_nan

    return blur_nan(a, sigma, torch)


# ---------------------------------------------------------------- 1. several passes, two models

def combine_passes(depths):
    """Depth maps of the same picture (NaN = unknown) from several passes -> (median depth, spread).

    Each pass is first brought to the first one's scale (median log ratio): a pass may come out a little
    nearer or farther as a whole. spread: per-pixel standard deviation of the passes in log depth."""
    logs = [log_depth(d) for d in depths]
    ref = logs[0]
    out = [ref]
    for lg in logs[1:]:
        both = np.isfinite(ref) & np.isfinite(lg)
        shift = float(np.median(ref[both] - lg[both])) if both.sum() > 100 else 0.0
        out.append(lg + shift)
    st = np.stack(out)
    n = np.isfinite(st).sum(0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(np.isfinite(st), st, np.nan), axis=0) if len(out) > 1 else ref
        spread = np.nanstd(st, axis=0) if len(out) > 1 else np.zeros_like(ref)
    med = np.where(n > 0, med, np.nan)
    return np.exp(med), np.where(n > 1, spread, np.nan)


def lens_choice(exif_fov, estimates, agree=0.08, off=0.15):
    """The horizontal field of view to use (degrees), where it comes from, and a note.

    ``estimates``: {model name: its own field of view of the photo (degrees) or None}. The EXIF lens is
    used unless the models agree with each other and both disagree with it in the same direction (a
    digitally zoomed or cropped photo whose EXIF still names the full lens); without EXIF, the models'
    mean when they agree, else the first model's own."""
    est = {k: float(v) for k, v in estimates.items() if v and 5 < float(v) < 170}
    vals = list(est.values())
    together = len(vals) >= 2 and abs(vals[0] - vals[1]) / (0.5 * (vals[0] + vals[1])) < agree
    if exif_fov:
        exif_fov = float(exif_fov)
        if together:
            mean = float(np.mean(vals))
            rel = [(v - exif_fov) / exif_fov for v in vals]
            if all(r > off for r in rel) or all(r < -off for r in rel):
                return mean, "models (EXIF disagreed)", (
                    f"the EXIF lens ({exif_fov:.1f}°) disagrees with both depth models ("
                    + ", ".join(f"{k} {v:.1f}°" for k, v in est.items())
                    + f"); {mean:.1f}° used (a zoomed or cropped photo?)")
        return exif_fov, "EXIF", "EXIF lens" + (" (confirmed by " + ", ".join(
            f"{k} {v:.1f}°" for k, v in est.items()) + ")" if est else "")
    if not vals:
        return None, "model", "lens estimated by the depth model"
    if len(vals) >= 2 and abs(vals[0] - vals[1]) / (0.5 * (vals[0] + vals[1])) < 2 * agree:
        mean = float(np.mean(vals[:2]))
        return mean, "models", "no EXIF: the depth models' mean lens (" + ", ".join(
            f"{k} {v:.1f}°" for k, v in est.items()) + ")"
    k0 = next(iter(est))
    return vals[0], "model", f"no EXIF: {k0}'s lens ({vals[0]:.1f}°)"


def depth_ensemble(primary, second, sigma, torch, w_primary=0.6, edge_rel=0.03, max_disagree=0.25):
    """Two depth maps of the same picture (metres, NaN = unknown) -> one, and stats.

    ``second`` is brought to ``primary``'s scale (median log ratio). Overall shape (blurred over ``sigma``
    pixels, log depth): the weighted mean of the two. Fine relief: their mean, except near depth edges,
    where ``second``'s (Depth Pro draws the sharpest edges). When the two disagree too much on the overall
    shape (median > ``max_disagree`` in log depth), ``primary`` alone is returned."""
    lp, ls = log_depth(primary), log_depth(second)
    both = np.isfinite(lp) & np.isfinite(ls)
    if both.sum() < 100:
        return np.asarray(primary, np.float64), {"used": False, "reason": "no overlap"}
    shift = float(np.median(lp[both] - ls[both]))
    ls = ls + shift
    lowp, lows = blur(lp, sigma, torch), blur(ls, sigma, torch)
    dis = np.abs(lowp - lows)
    med = float(np.median(dis[both]))
    stats = {"scale_second_to_primary": round(float(np.exp(shift)), 4), "disagreement_median": round(med, 4),
             "disagreement_p90": round(float(np.quantile(dis[both], 0.9)), 4)}
    if med > max_disagree:
        stats.update(used=False, reason="the two models disagree too much on the overall shape")
        return np.asarray(primary, np.float64), stats
    hp, hs = lp - lowp, ls - lows
    gx = np.abs(np.diff(ls, axis=1, append=ls[:, -1:]))
    gy = np.abs(np.diff(ls, axis=0, append=ls[-1:, :]))
    edge = np.nan_to_num((gx > edge_rel) | (gy > edge_rel)).astype(bool)
    edge = _dilate(edge, 2)
    low = np.where(both, w_primary * lowp + (1 - w_primary) * lows, lowp)
    hf = np.where(both, np.where(edge, hs, 0.5 * (hp + hs)), hp)
    out = np.where(np.isfinite(lp), np.exp(low + hf), np.nan)
    stats.update(used=True, edge_share=round(float(edge[both].mean()), 4))
    return out, stats


def _dilate(m, r):
    m = np.asarray(m, bool)
    for _ in range(int(r)):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


# ---------------------------------------------------------------- 3. normals -> fine relief

def normals_from_depth(depth, fx, cx=None, cy=None):
    """Surface normals (H, W, 3, unit, facing the camera) of a depth map, from its 3D points; NaN where
    unknown."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    p = rays(h, w, fx, fx, cx, cy) * d[..., None]
    du = np.full_like(p, np.nan)
    dv = np.full_like(p, np.nan)
    du[:, 1:-1] = p[:, 2:] - p[:, :-2]
    dv[1:-1] = p[2:] - p[:-2]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    flip = (n * p).sum(-1) > 0                  # facing the camera: n . p < 0
    return np.where(flip[..., None], -n, n)


def align_normals(pred, ref, weight=None, max_angle=30.0):
    """The orthogonal map Q (rotation, or a reflection: axis conventions differ between models) that best
    turns predicted normals into the depth's own normals, and how well they then agree.

    Returns (aligned normals (H, W, 3), {"median_angle_deg", "ok", "matrix"}). Fine relief is absent from
    the depth's normals but averages out in the fit; ok = median angle below ``max_angle``."""
    a = np.asarray(pred, np.float64)
    b = np.asarray(ref, np.float64)
    ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    if weight is not None:
        ok &= np.asarray(weight) > 0.5
    if ok.sum() < 200:
        return a, {"ok": False, "reason": "too few pixels", "median_angle_deg": None}
    A, B = a[ok], b[ok]
    M = B.T @ A                                  # sum of b a^T
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    out = a @ Q.T
    cos = np.clip(np.abs((out[ok] * B).sum(-1)), -1, 1)
    ang = float(np.degrees(np.median(np.arccos(cos))))
    return out, {"ok": ang < max_angle, "median_angle_deg": round(ang, 2),
                 "matrix": [[round(float(x), 4) for x in r] for r in Q]}


def normal_slopes(normals, fx, cx=None, cy=None, max_slope=0.08, min_facing=0.15):
    """Slopes of log depth along u and v (H, W each) that surface normals imply: for a point z * r(u, v),
    n . dP/du = 0 gives d ln z / du = -n_x / (f n . r) (and n_y for v). NaN where the surface is seen at a
    grazing angle (|n . r| / |r| < ``min_facing``); clamped to ``max_slope`` per pixel."""
    n = np.asarray(normals, np.float64)
    h, w = n.shape[:2]
    r = rays(h, w, fx, fx, cx, cy)
    nr = (n * r).sum(-1)
    facing = np.abs(nr) / np.linalg.norm(r, axis=-1)
    with np.errstate(all="ignore"):
        su = np.clip(-n[..., 0] / (fx * nr), -max_slope, max_slope)
        sv = np.clip(-n[..., 1] / (fx * nr), -max_slope, max_slope)
    bad = ~np.isfinite(nr) | (facing < min_facing)
    return np.where(bad, np.nan, su), np.where(bad, np.nan, sv)


def integrate_normals(depth, normals, fx, torch, weight=None, cutoff=None, limit=0.05, iters=300, device=None,
                      cx=None, cy=None):
    """Fine relief from surface normals joined to a depth map (metres, NaN = unknown).

    Solves for the log depth L closest to the slopes the normals imply and, at long wavelengths, to the
    given depth (screened Poisson, conjugate gradients): sum w (L_q - L_p - s_pq)^2 + lam sum (L - L0)^2,
    lam = (2 pi / cutoff)^2: relief finer than ``cutoff`` pixels comes from the normals, the shape and
    scale from the depth. No slope is used across depth edges or where ``weight`` (H, W in [0, 1]) is
    low. The change is capped at ``limit`` (log units) so a normal model's mistake cannot warp the
    scene. Returns (depth, stats)."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    L0 = log_depth(d)
    ok = np.isfinite(L0)
    cutoff = cutoff or max(8.0, 0.01 * max(h, w))
    lam = (2 * np.pi / cutoff) ** 2
    su, sv = normal_slopes(normals, fx, cx, cy)
    wt = np.ones((h, w)) if weight is None else np.clip(np.nan_to_num(np.asarray(weight, np.float64)), 0, 1)
    # an edge between two pixels counts when both are known, have a slope and are not across a depth edge
    gx = np.abs(np.diff(np.nan_to_num(L0), axis=1))
    gy = np.abs(np.diff(np.nan_to_num(L0), axis=0))
    wu = np.minimum(wt[:, 1:], wt[:, :-1]) * (ok[:, 1:] & ok[:, :-1]) * (gx < 0.05)
    wv = np.minimum(wt[1:], wt[:-1]) * (ok[1:] & ok[:-1]) * (gy < 0.05)
    tu = 0.5 * (su[:, 1:] + su[:, :-1])
    tv = 0.5 * (sv[1:] + sv[:-1])
    wu = np.where(np.isfinite(tu), wu, 0.0)
    wv = np.where(np.isfinite(tv), wv, 0.0)
    if wu.sum() + wv.sum() < 100:
        return d, {"used": False, "reason": "no usable normals"}
    dev = device or torch.device("cpu")
    T = lambda a: torch.as_tensor(np.nan_to_num(np.asarray(a, np.float64)), dtype=torch.float32, device=dev)  # noqa
    L0t, Wu, Wv = T(np.where(ok, L0, 0.0)), T(wu), T(wv)
    Tu, Tv = T(tu), T(tv)
    C = T(ok.astype(np.float64)) * lam + (1 - T(ok.astype(np.float64))) * 1.0   # unknown pixels: held at 0

    def Dt_W(ru, rv):                       # D^T applied to edge residuals (u: w-1 columns, v: h-1 rows)
        out = torch.zeros((h, w), device=dev)
        out[:, 1:] += ru
        out[:, :-1] -= ru
        out[1:] += rv
        out[:-1] -= rv
        return out

    def A(x):                                # (D^T W D + C) x
        ru = Wu * (x[:, 1:] - x[:, :-1])
        rv = Wv * (x[1:] - x[:-1])
        return Dt_W(ru, rv) + C * x

    # solve for the change delta = L - L0
    b = Dt_W(Wu * (Tu - (L0t[:, 1:] - L0t[:, :-1])), Wv * (Tv - (L0t[1:] - L0t[:-1])))
    diag = C.clone()
    diag[:, 1:] += Wu
    diag[:, :-1] += Wu
    diag[1:] += Wv
    diag[:-1] += Wv
    Minv = 1.0 / diag.clamp(min=1e-8)
    x = torch.zeros((h, w), device=dev)
    r = b - A(x)
    z = Minv * r
    p = z.clone()
    rz = (r * z).sum()
    b_norm = float(b.norm()) + 1e-12
    it = 0
    for it in range(1, iters + 1):
        Ap = A(p)
        alpha = rz / (p * Ap).sum().clamp(min=1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        if float(r.norm()) / b_norm < 1e-4:
            break
        z = Minv * r
        rz_new = (r * z).sum()
        p = z + (rz_new / rz.clamp(min=1e-20)) * p
        rz = rz_new
    delta = x.cpu().numpy().astype(np.float64)
    clamped = float((np.abs(delta[ok]) > limit).mean()) if ok.any() else 0.0
    delta = np.clip(delta, -limit, limit)
    out = np.where(ok, np.exp(L0 + delta), np.nan)
    return out, {"used": True, "cutoff_px": round(float(cutoff), 1), "iterations": it,
                 "rms_change": round(float(np.sqrt(np.mean(delta[ok] ** 2))), 5), "capped_share": round(clamped, 4),
                 "slope_share": round(float((wu > 0.5).mean()), 4)}


def blend_normals(shape, parts):
    """Normal maps of overlapping parts [((x0, y0, x1, y1), normals (ch, cw, 3))] -> one (H, W, 3) map,
    feathered at the parts' borders; NaN where no part has a normal."""
    from detail_geometry import feather

    H, W = shape
    acc = np.zeros((H, W, 3))
    ws = np.zeros((H, W))
    for (x0, y0, x1, y1), n in parts:
        n = np.asarray(n, np.float64)
        f = feather(x1 - x0, y1 - y0) * np.isfinite(n).all(-1)
        acc[y0:y1, x0:x1] += np.nan_to_num(n) * f[..., None]
        ws[y0:y1, x0:x1] += f
    out = acc / np.maximum(np.linalg.norm(acc, axis=-1, keepdims=True), 1e-9)
    return np.where((ws > 1e-6)[..., None], out, np.nan)


def resize_normals(n, size, torch):
    """An (h, w, 3) normal map resized to size = (H, W), renormalised; NaN where unknown."""
    from detail_geometry import resize_nan

    out = np.stack([resize_nan(n[..., i], size, torch) for i in range(3)], -1)
    return out / np.maximum(np.linalg.norm(np.nan_to_num(out), axis=-1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- 4. clean depth edges

def _extend(vals, known, iters, F, need=None):
    """Values (1, C, H, W) known where ``known`` (1, 1, H, W) pushed outwards (mean of known neighbours),
    until every pixel of ``need`` has one or after ``iters`` steps."""
    v = vals * known
    m = known.clone()
    for _ in range(int(iters)):
        if need is not None and bool(((m > 0) | ~need).all()):
            break
        num = F.avg_pool2d(v, 3, 1, 1, count_include_pad=False)
        den = F.avg_pool2d(m, 3, 1, 1, count_include_pad=False)
        grow = ((m == 0) & (den > 0)).float()
        v = v + grow * (num / den.clamp(min=1e-9))
        m = m + grow
    return v, m


def resolve_edges(depth, img, torch, rel=0.04, radius=3, device=None):
    """Depth near depth edges sent to one side: the near surface or the far one, never in between.

    Around every depth step larger than ``rel`` (log depth between neighbours), within ``radius`` pixels
    (the band): the near and the far surface are the depths just outside the band, on each side (split at
    the middle of the local depth range), carried into it; each band pixel takes the depth of the side
    whose colour it has (its own depth breaks ties). Returns (depth, band (H, W bool: changed pixels), far
    depth (H, W: the far side's depth in the band, for the splats behind), stats)."""
    import torch.nn.functional as F

    dev = device or torch.device("cpu")
    L = log_depth(depth)
    ok = np.isfinite(L)
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    with np.errstate(invalid="ignore"):
        gx[:, :-1] = np.abs(np.diff(L, axis=1))
        gy[:-1] = np.abs(np.diff(L, axis=0))
    step = (np.nan_to_num(gx) > rel) | (np.nan_to_num(gy) > rel)
    band = _dilate(step, radius) & ok
    if band.sum() == 0:
        return np.asarray(depth, np.float64), band, np.full(L.shape, np.nan), {"band_share": 0.0}
    r2 = 2 * radius + 1
    k = 2 * r2 + 1
    Lt = torch.as_tensor(np.where(ok, L, 0.0), dtype=torch.float32, device=dev)[None, None]
    okt = torch.as_tensor(ok.astype(np.float32), device=dev)[None, None]
    bandt = torch.as_tensor(band, device=dev)[None, None]
    big = 1e4
    near = -F.max_pool2d(-(Lt + (1 - okt) * big), k, 1, r2)
    far = F.max_pool2d(Lt - (1 - okt) * big, k, 1, r2)
    mid = 0.5 * (near + far)
    outside = (okt > 0) & ~bandt & ((far - near) > rel)      # just outside the band, the step in sight
    fgk = (outside & (Lt < mid)).float()
    bgk = (outside & (Lt >= mid)).float()
    reach = 6 * radius + 8
    lf, mf = _extend(Lt, fgk, reach, F, bandt)
    lb, mb = _extend(Lt, bgk, reach, F, bandt)
    im = torch.as_tensor(np.asarray(img, np.float32) / 255.0, device=dev).permute(2, 0, 1)[None]
    cf, _ = _extend(im, fgk, reach, F, bandt)
    cb, _ = _extend(im, bgk, reach, F, bandt)
    dcf = (im - cf).norm(dim=1, keepdim=True)
    dcb = (im - cb).norm(dim=1, keepdim=True)
    s = (dcf - dcb) / (dcf + dcb + 1e-3)                 # < 0: the colour of the near side
    t = (Lt - 0.5 * (lf + lb)) / (lb - lf).abs().clamp(min=1e-6)      # < 0: nearer than the middle
    to_fg = (0.7 * s + 0.6 * t) < 0
    real = bandt & (mf > 0) & (mb > 0) & ((lb - lf) > rel)
    new = torch.where(to_fg, lf, lb)
    out = torch.where(real, new, Lt)[0, 0].cpu().numpy().astype(np.float64)
    band2 = real[0, 0].cpu().numpy()
    far_d = np.where(band2, np.exp(lb[0, 0].cpu().numpy().astype(np.float64)), np.nan)
    res = np.where(ok, np.exp(out), np.nan)
    return res, band2, far_d, {"band_share": round(float(band2.mean()), 4),
                               "to_near_share": round(float(to_fg[real].float().mean()) if real.any() else 0.0, 3)}
