"""Maximum detail for Real 3D photos (2.0.0-rc6): more of the small geometry of a photo.

Three limits of the normal path, and what this module does about each:
  1. MoGe-3 sees the photo at most 1536 px (and works internally at less): small relief comes out smooth.
     -> patch fusion: MoGe-3 on overlapping crops of the photo at up to 3072 px, each aligned to the
        whole-photo depth; the whole photo gives the overall shape and scale (low frequencies), the crops
        the fine relief (high frequencies). Same idea as Boosting Monocular Depth / PatchFusion.
  2. SHARP follows the injected depth only through a smooth learned alignment.
     -> every splat the photo sees is placed at that depth (sharp_depth_worker.snap_to_depth), with little
        weight near depth edges where a pixel's depth is ambiguous.
  3. SHARP makes 2 x 768 x 768 splats for the whole photo: one per ~5 x 5 pixels of a 4000 px photo.
     -> SHARP on overlapping crops (2 x 2; 3 x 3 for photos of 24 MP and up, rc7), each with its real camera geometry (off-centre principal
        point), each crop keeping the splats of the part it owns; nearly transparent splats are dropped.
numpy + torch only (runs in the 'photo-cu128' runtime, inside sharp_depth_worker).
"""

import numpy as np

WORK_SIDE = 3072          # the detail path works on the photo at up to this many pixels (long side)
ULTRA_SIDE = 4608         # rc7: ... and photos of ULTRA_FROM px and more (24 MP and up) at this many, 3 x 3 parts
ULTRA_FROM = 5000


def work_side(photo_side):
    """The long side the detail path works at for a photo whose long side is ``photo_side`` pixels."""
    return ULTRA_SIDE if photo_side >= ULTRA_FROM else WORK_SIDE


def tile_count(w, h, work_side=WORK_SIDE):
    """Crops per side: 2 when the photo has more pixels than SHARP / MoGe take in one go, 3 beyond 3600 px
    (rc7: 24 MP photos and up, worked at 4608 px), else 1."""
    m = max(w, h)
    return 3 if m > 3600 else 2 if m > 2048 else 1


def tile_boxes(w, h, n, overlap=0.2):
    """n x n overlapping crops covering a (w, h) picture: [((x0, y0, x1, y1) crop, (x0, y0, x1, y1) core)];
    the cores (the part each crop owns) tile the picture exactly."""
    xs, ys = np.linspace(0, w, n + 1).round().astype(int), np.linspace(0, h, n + 1).round().astype(int)
    ox, oy = int(overlap * w / n), int(overlap * h / n)
    out = []
    for j in range(n):
        for i in range(n):
            core = (int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1]))
            crop = (max(0, core[0] - ox), max(0, core[1] - oy), min(w, core[2] + ox), min(h, core[3] + oy))
            out.append((crop, core))
    return out


def resize_nan(a, size, torch):
    """An (h, w) map with NaN where unknown, resized to size = (H, W) without letting unknown pixels leak."""
    import torch.nn.functional as F

    t = torch.as_tensor(np.asarray(a, np.float32))
    ok = torch.isfinite(t)
    num = F.interpolate(torch.where(ok, t, 0.0)[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    den = F.interpolate(ok.float()[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    out = num / den.clamp(min=1e-6)
    return torch.where(den > 0.5, out, torch.nan).numpy().astype(np.float64)


def blur_nan(a, sigma, torch):
    """Gaussian blur of an (H, W) map that ignores NaN (normalised convolution); NaN stays NaN."""
    import torch.nn.functional as F

    t = torch.as_tensor(np.asarray(a, np.float32))
    ok = torch.isfinite(t)
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()

    def conv(m):
        m = F.conv2d(m[None, None], k.view(1, 1, 1, -1), padding=(0, r))
        return F.conv2d(m, k.view(1, 1, -1, 1), padding=(r, 0))[0, 0]

    num, den = conv(torch.where(ok, t, 0.0)), conv(ok.float())
    return torch.where(ok, num / den.clamp(min=1e-6), torch.nan).numpy().astype(np.float64)


def feather(cw, ch):
    """Weights of a crop for blending: 1 in the middle falling linearly to ~0 at its border."""
    wx = np.minimum(np.arange(cw) + 1, cw - np.arange(cw)) / (cw / 2)
    wy = np.minimum(np.arange(ch) + 1, ch - np.arange(ch)) / (ch / 2)
    return np.clip(np.outer(wy, wx), 1e-3, 1.0)


def patch_fusion(global_depth, crops, sigma, torch, limit=0.08):
    """Whole-photo depth (H, W, metres, NaN = unknown) + crop depths [((x0, y0, x1, y1), depth (ch, cw))] ->
    one depth with the whole photo's low frequencies (overall shape, scale) and the crops' high frequencies
    (fine relief), in log depth. Each crop is first aligned (robust affine) to the whole-photo depth.
    ``limit`` caps the added relief (log units) so a crop's mistake cannot make spikes.
    Returns (fused depth, stats)."""
    from sharp_depth_worker import align_affine

    g = np.asarray(global_depth, np.float64)
    ok = np.isfinite(g) & (g > 0)
    lg = np.where(ok, np.log(np.where(ok, g, 1.0)), np.nan)
    low = blur_nan(lg, sigma, torch)
    hp = np.zeros_like(g)
    wsum = np.zeros_like(g)
    stats = []
    for (x0, y0, x1, y1), d in crops:
        d = np.asarray(d, np.float64)
        ref = g[y0:y1, x0:x1]
        both = np.isfinite(d) & (d > 0) & np.isfinite(ref) & (ref > 0)
        if both.sum() < 100:
            continue
        a, b = align_affine(d[both], ref[both])
        da = a * d + b
        dok = np.isfinite(da) & (da > 0)
        ld = np.where(dok, np.log(np.where(dok, da, 1.0)), np.nan)
        h = ld - blur_nan(ld, sigma, torch)
        f = feather(x1 - x0, y1 - y0) * np.isfinite(h)
        hp[y0:y1, x0:x1] += np.nan_to_num(h) * f
        wsum[y0:y1, x0:x1] += f
        stats.append({"box": [int(x0), int(y0), int(x1), int(y1)], "scale": round(a, 4), "shift": round(b, 4)})
    detail = np.clip(hp / np.maximum(wsum, 1e-6), -limit, limit)
    fused = np.where(ok, np.exp(low + np.where(wsum > 1e-3, detail, lg - low)), np.nan)
    return fused, stats


def edge_weight(depth, torch, rel=0.03, grow=3, soften=2.0):
    """Confidence that a pixel's depth belongs to one surface: ~0 near depth edges (a jump of more than
    ``rel`` between neighbouring pixels) and where the depth is unknown, 1 elsewhere."""
    d = np.asarray(depth, np.float64)
    ok = np.isfinite(d) & (d > 0)
    ld = np.where(ok, np.log(np.where(ok, d, 1.0)), 0.0)
    gx = np.abs(np.diff(ld, axis=1, append=ld[:, -1:]))
    gy = np.abs(np.diff(ld, axis=0, append=ld[-1:, :]))
    edge = (gx > rel) | (gy > rel) | ~ok
    for _ in range(grow):
        p = np.pad(edge, 1)
        edge = edge | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    w = (~edge).astype(np.float64)
    return np.clip(np.nan_to_num(blur_nan(w, soften, torch)), 0, 1) * ok


def owned(crop, core, g=768):
    """Which of a crop's g x g SHARP grid cells lie in the part of the picture the crop owns: (g, g) bool."""
    x0, y0, x1, y1 = crop
    cx0, cy0, cx1, cy1 = core
    u = x0 + (np.arange(g) + 0.5) / g * (x1 - x0)
    v = y0 + (np.arange(g) + 0.5) / g * (y1 - y0)
    return ((v >= cy0) & (v < cy1))[:, None] & ((u >= cx0) & (u < cx1))[None, :]
