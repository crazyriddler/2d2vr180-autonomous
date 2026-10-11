"""4D reconstruction of a video (runtime 'recon3d-cu124').

All frames at a fixed rate →
  VGGT camera poses + depth for every frame (chunked, Sim(3)-stitched; a camera that does not move is
  held exactly still) → metric scale (MoGe-2) →
  MoGe-2 depth of every frame, aligned (scale + shift) onto VGGT's joint geometry →
  what moves: every frame is reprojected into the frames 0.5, 1.5 and 3 s before and after it; a pixel
  whose 3D point lies in front of the surface the other frame sees there (free space), or has another
  colour at the same depth, is moving →
  the depth of every frame is aligned again on its still pixels only (movers no longer bias it) →
  background: one 3D Gaussian splat trained on all frames with the moving pixels left out, so what a
  passer-by hides in one frame is filled in from the others →
  moving parts: Gaussians that live for a while and move (motion_trainer.py), seeded from the moving
  splats of every second frame and optimised on all frames in front of the background - smooth at any
  frame rate; if that fails, per-frame splats visible only at their frame's time (front surface: what
  the camera saw) →
  background.ply (+ provenance), dynamic.npz (scene4d.DynamicSplats layout), cameras.json (one camera
  per frame), a TSDF mesh of the background and a preview of what was found to move.

Request:
  {"frames": ["frame_00001.png", ...], "fps": 12.0, "output_dir": "...", "vggt_dir": "...",
   "moge_path": ".../model.pt", "max_side": 960, "steps": 9000, "max_gaussians": 1500000,
   "fg_budget": 40000, "camera": "auto" | "fixed" | "moving", "mask_offsets_s": [0.5, 1.5, 3.0],
   "motion": true, "motion_steps": 4000, "keyframe_stride": 2, "mesh": true}
Coordinates: OpenCV, scene frame = the first frame's camera.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

MASK_SIDE = 320          # motion is detected at this resolution
INFERRED, OBSERVED = 1, 0


# ------------------------------------------------------------------ geometry helpers (numpy only)
def project(X, K, c2w):
    """World points (..., 3) → pixel u, v and depth z in a camera."""
    import numpy as np

    w2c = np.linalg.inv(c2w)
    pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[..., 0] / z + K[0, 2]
        v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    return u, v, z


def unproject(depth, K, c2w):
    import numpy as np

    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * depth, (ys + 0.5 - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def scale_K(K, sx, sy):
    import numpy as np

    K = np.array(K, np.float64)
    K[0] *= sx
    K[1] *= sy
    return K


def depth_edges(z, rel=0.04):
    """Pixels on a depth discontinuity (relative jump to a 4-neighbour): monocular depth puts 'flying
    pixels' at interpolated depths there."""
    import numpy as np

    e = np.zeros(z.shape, bool)
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                 ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        za, zb = z[a], z[b]
        with np.errstate(invalid="ignore"):
            jump = np.abs(za - zb) > rel * np.minimum(za, zb)
        e[a] |= jump
        e[b] |= jump
    return e


def camera_is_fixed(c2ws, depth_median, trans_tol=0.01, rot_tol_deg=1.0):
    """True when the camera neither moves (relative to the scene's depth) nor turns."""
    import numpy as np

    R0, t0 = c2ws[0][:3, :3], c2ws[0][:3, 3]
    spread = max(float(np.linalg.norm(m[:3, 3] - t0)) for m in c2ws)
    ang = max(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ m[:3, :3]) - 1) / 2)))) for m in c2ws)
    return spread <= trans_tol * max(depth_median, 1e-6) and ang <= rot_tol_deg


def rotmat_to_quat(R):
    """(w, x, y, z) of a rotation matrix."""
    import numpy as np

    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
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
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


# ------------------------------------------------------------------ what moves
FREE_SPACE, CHANGED, STILL, NO_EVIDENCE = 2, 1, -1, 0


def motion_votes(img_i, z_i, K_i, c2w_i, img_j, z_j, K_j, c2w_j, tau=0.08, free_tau=0.15, colour_thr=0.12):
    """Evidence from frame j about every pixel of frame i: FREE_SPACE (j sees a surface clearly behind
    where i's point would be: at j's time that place is empty, so what i sees there is not permanent),
    CHANGED (same depth, another colour, after removing the overall exposure difference between the
    frames), STILL (same depth and colour) or NO_EVIDENCE (hidden behind something in j, outside j's
    view, or on a depth edge of i, where monocular depth is unreliable)."""
    import numpy as np

    h, w = z_i.shape
    valid = np.isfinite(z_i) & (z_i > 0) & ~depth_edges(np.where(z_i > 0, z_i, np.inf))
    X = unproject(np.where(valid, z_i, 1.0), K_i, c2w_i)
    u, v, z = project(X, K_j, c2w_j)
    hj, wj = z_j.shape
    inb = valid & np.isfinite(u) & np.isfinite(v) & (z > 1e-6) & (u >= 0) & (u < wj) & (v >= 0) & (v < hj)
    ui = np.where(inb, u, 0).astype(np.int64)
    vi = np.where(inb, v, 0).astype(np.int64)
    zj = z_j[vi, ui]
    ok = inb & np.isfinite(zj) & (zj > 0)
    front = ok & (z < zj * (1 - free_tau))
    same = ok & (np.abs(z - zj) <= tau * zj)
    ci = img_i.astype(np.float32) / 255
    cj = img_j[vi, ui].astype(np.float32) / 255
    off = np.median((cj - ci)[same], axis=0) if same.sum() > 100 else np.zeros(3, np.float32)
    # best match within one pixel: reprojection is never exact, and sharp texture would read as change
    diff = np.full((h, w), np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = img_j[np.clip(vi + dy, 0, hj - 1), np.clip(ui + dx, 0, wj - 1)].astype(np.float32) / 255
            diff = np.minimum(diff, np.abs(c - off - ci).mean(-1))
    votes = np.full((h, w), NO_EVIDENCE, np.int8)
    votes[same & (diff <= colour_thr)] = STILL
    votes[same & (diff > colour_thr)] = CHANGED
    votes[front] = FREE_SPACE
    return votes


def clean_mask(m, min_frac=0.0015, grow_frac=0.012):
    """Remove specks and small blobs, close gaps, fill holes inside movers and grow the mask a little
    (edges, motion blur, shadows)."""
    import numpy as np

    try:
        from scipy import ndimage
    except ImportError:          # the runtime ships scipy (scikit-learn depends on it); raw mask otherwise
        return m
    h, w = m.shape
    m = np.pad(m, 3, mode="edge")             # scipy's closing erodes along the image border otherwise
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)[3:-3, 3:-3]
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_frac * h * w
        m = keep[lab]
    m = ndimage.binary_fill_holes(m)
    r = max(1, int(round(grow_frac * max(h, w))))
    return ndimage.binary_dilation(m, iterations=r)


def neighbours(i, n, fps, offsets_s=(0.5, 1.5, 3.0), anchors=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Frames compared with frame i: some before and after it, and a few spread over the whole clip (so
    someone who stands still for a while and then leaves is still found to be passing through)."""
    steps = sorted({max(1, int(round(o * fps))) for o in offsets_s})
    js = {j for d in steps for j in (i - d, i + d) if 0 <= j < n}
    js |= {int(round(a * (n - 1))) for a in anchors}
    js.discard(i)
    return sorted(js)


def moving_masks(imgs, depths, Ks, c2ws, fps, offsets_s=(0.5, 1.5, 3.0), report=None):
    """Per frame, a bool mask of what is not permanent (images and depths at one common resolution): a
    pixel moves if any compared frame sees free space where its point is, or if most frames that see
    the same surface see another colour there."""
    import numpy as np

    n = len(imgs)
    out = []
    for i in range(n):
        if report:
            report(i, n)
        free = np.zeros(depths[i].shape, bool)
        changed = np.zeros(depths[i].shape, np.int16)
        still = np.zeros(depths[i].shape, np.int16)
        for j in neighbours(i, n, fps, offsets_s):
            v = motion_votes(imgs[i], depths[i], Ks[i], c2ws[i], imgs[j], depths[j], Ks[j], c2ws[j])
            free |= v == FREE_SPACE
            changed += v == CHANGED
            still += v == STILL
        out.append(clean_mask(free | ((changed > 0) & (changed >= still))))
    return out


# ------------------------------------------------------------------ moving splats
def frame_splats(img, z, mask, K, c2w, budget, footprint=0.75, thickness=0.25):
    """Splats for the moving pixels of one frame: on a regular pixel grid (stride chosen to fit
    ``budget``), each a flat disc facing the camera and as wide as the pixels it stands for."""
    import numpy as np

    ok = mask & np.isfinite(z) & (z > 0)
    ok &= ~depth_edges(np.where(ok, z, np.inf))
    n = int(ok.sum())
    if n == 0:
        return None
    stride = max(1, int(math.ceil(math.sqrt(n / max(budget, 1)))))
    grid = np.zeros_like(ok)
    grid[stride // 2::stride, stride // 2::stride] = True
    ys, xs = np.nonzero(ok & grid)
    if len(ys) == 0:
        return None
    zz = z[ys, xs].astype(np.float64)
    pc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * zz, (ys + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], -1)
    X = pc @ c2w[:3, :3].T + c2w[:3, 3]
    s = footprint * stride * zz / ((K[0, 0] + K[1, 1]) / 2)
    m = len(X)
    return {"means": X.astype(np.float32),
            "scales": np.stack([s, s, s * thickness], 1).astype(np.float32),
            "quats": np.tile(rotmat_to_quat(c2w[:3, :3]).astype(np.float32), (m, 1)),
            "opacity": np.full(m, 0.95, np.float32),
            "colors": (img[ys, xs].astype(np.float32) / 255),
            "stride": stride}


def resize_img(img, w, h):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def resize_nearest(m, w, h):
    import numpy as np

    sh, sw = m.shape[:2]
    yi = np.minimum(((np.arange(h) + 0.5) * sh / h).astype(int), sh - 1)
    xi = np.minimum(((np.arange(w) + 0.5) * sw / w).astype(int), sw - 1)
    return m[yi][:, xi]


def mask_preview(imgs, masks, path, count=6):
    import numpy as np
    from PIL import Image

    idx = np.linspace(0, len(imgs) - 1, min(count, len(imgs))).round().astype(int)
    tiles = []
    for i in idx:
        t = imgs[i].astype(np.float32).copy()
        t[masks[i]] = t[masks[i]] * 0.4 + np.array([255, 40, 40]) * 0.6
        tiles.append(t.astype(np.uint8))
    Image.fromarray(np.concatenate(tiles, 1)).save(path)


def moving_gaussians(bg_params, train_imgs, c2ws, Ks_train, Ks, imgs, depth, moving, n, fps, budget, req, torch):
    """The moving parts as Gaussians that live for a while and move (motion_trainer), seeded from the moving
    splats of every few frames and optimised on all frames in front of the trained background."""

    import numpy as np

    import motion_trainer as mt

    cfg = mt.MotionConfig(steps=int(req.get("motion_steps", 4000)),
                          keyframe_stride=max(1, int(req.get("keyframe_stride", 2))))
    keys = list(range(0, n, cfg.keyframe_stride))
    if keys[-1] != n - 1:
        keys.append(n - 1)
    kfs = []
    for k in keys:
        sp = frame_splats(imgs[k], depth[k], moving[k], Ks[k], c2ws[k], budget)
        if sp is not None:
            kfs.append({**sp, "t": (k + 0.5) / n})
    if not kfs:
        raise ValueError("no moving pixels with depth")
    progress(0.88, f"moving parts in 4D: following {len(kfs)} keyframes")
    seed, graph = mt.seed_motion(kfs, cfg, max_speed=float(req.get("max_speed_m_s", 15.0)) * n / fps)
    params, t_center = mt.to_params(seed, "cuda")
    log(f"moving parts in 4D: {len(seed['means']):,} Gaussians from {len(kfs)} keyframes")
    tw, th = train_imgs[0].shape[1], train_imgs[0].shape[0]
    masks = [resize_nearest(m, tw, th) for m in moving]
    deps = [resize_nearest(d, tw, th) for d in depth]
    bg = {k: bg_params[k].detach() for k in ("means", "quats", "scales", "opacities", "sh0", "shN")}
    hist = mt.train_motion(params, t_center, graph, train_imgs, c2ws, Ks_train, [(i + 0.5) / n for i in range(n)],
                           masks, deps, cfg, background=bg, device="cuda",
                           progress=lambda v, m: progress(0.88 + 0.06 * v, m))
    out = mt.export_motion(params, t_center)
    first = float(np.mean([h[1] for h in hist[:4]]))
    last = float(np.mean([h[1] for h in hist[-4:]]))
    log(f"moving parts in 4D: {len(out['means']):,} Gaussians kept; loss {first:.4f} → {last:.4f}")
    return out


# ------------------------------------------------------------------ worker
def main(req):
    import numpy as np
    import torch

    import multiview_worker as mv
    import splat_trainer as st

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="4D video reconstruction needs an NVIDIA GPU (CUDA).")
        sys.exit(1)
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    paths = req["frames"]
    n = len(paths)
    if n < 2:
        emit("error", code="bad_input", message="A 4D reconstruction needs at least 2 frames.")
        sys.exit(1)
    fps = float(req.get("fps", 12.0))
    device = "cuda"
    progress(0.01, f"loading {n} frames")
    imgs = [mv.load_rgb(p, int(req.get("max_side", 960))) for p in paths]
    h, w = imgs[0].shape[:2]
    vin = [mv.vggt_input(im) for im in imgs]

    # ------------------------------------------------------------ cameras
    from vggt.models.vggt import VGGT

    progress(0.03, "loading VGGT")
    vdir = req["vggt_dir"]
    if os.path.exists(os.path.join(vdir, "config.json")) and os.path.exists(os.path.join(vdir, "model.safetensors")):
        model = VGGT.from_pretrained(vdir)
    else:
        model = VGGT()
        model.load_state_dict(torch.load(os.path.join(vdir, "model.pt"), map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    chunk = int(req.get("chunk") or (24 if env.get("vram_total_mib", 16000) / 1024 >= 15 else 12))
    c2w, K_v, vdepth, vconf = mv.estimate_cameras(vin, model, dtype, torch, chunk, int(req.get("overlap", 8)))
    del model
    mv.free(torch)
    med = float(np.median([np.median(d[d > 0]) for d in vdepth if (d > 0).any()]))
    hint = req.get("camera", "auto")
    fixed = hint == "fixed" or (hint == "auto" and camera_is_fixed(c2w, med))
    if fixed:            # a camera on a tripod: hold it exactly still (no pose jitter)
        c2w = [np.eye(4) for _ in range(n)]
        log("the camera does not move: all frames share one camera")
    else:
        log("the camera moves: one camera pose per frame")

    scale = None
    if req.get("moge_path"):
        progress(0.3, "metric scale (MoGe-2)")
        try:
            scale = mv.metric_scale(req["moge_path"], vin, vdepth, vconf, torch, max_views=6, device=device)
        except Exception as e:  # noqa: BLE001 - metric scale is optional
            log(f"metric scale failed: {e}")
    if scale and np.isfinite(scale) and scale > 0:
        log(f"metric scale factor {scale:.4f}")
        vdepth = [d * scale for d in vdepth]
        for m in c2w:
            m[:3, 3] *= scale
    else:
        scale = None
    Ks = [mv.to_original_K(K_v[i], vin[i][1], 0) for i in range(n)]
    if fixed:            # one lens for all frames too
        f = float(np.median([K[0, 0] for K in Ks]))
        K0 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        Ks = [K0.copy() for _ in range(n)]

    # ------------------------------------------------------------ depth of every frame
    from moge.model.v2 import MoGeModel

    moge = MoGeModel.from_pretrained(req["moge_path"]).to(device).eval()
    raw, zv_all, trusted = [], [], []
    for i in range(n):
        progress(0.32 + 0.12 * i / n, f"depth of every frame (MoGe-2) {i + 1}/{n}")
        t = torch.from_numpy(imgs[i]).to(device).float().div(255).permute(2, 0, 1)
        fov = math.degrees(2 * math.atan(w / 2 / Ks[i][0, 0]))
        with torch.no_grad():
            res = moge.infer(t, resolution_level=9, use_fp16=True, fov_x=fov)
        zm = res["depth"].float().cpu().numpy()
        ok = np.isfinite(zm) & (zm > 0)
        if "mask" in res:
            ok &= res["mask"].cpu().numpy().astype(bool)
        raw.append(np.where(ok, zm, 0).astype(np.float32))
        zv, inside = mv.vggt_on_image(vdepth[i], vin[i][1], h, w)
        cv, _ = mv.vggt_on_image(vconf[i], vin[i][1], h, w)
        zv_all.append(np.where(inside, zv, 0).astype(np.float32))
        trusted.append(inside & (cv >= np.percentile(vconf[i], 50)))
        del res, t
    del moge
    mv.free(torch)

    def aligned(i, use):
        fit = mv.align_affine(raw[i], zv_all[i], use & (raw[i] > 0))
        if fit is None:
            return np.zeros_like(raw[i])
        z = raw[i] * fit[0] + fit[1]
        return np.where((raw[i] > 0) & (z > 0), z, 0).astype(np.float32)

    depth = [aligned(i, trusted[i]) for i in range(n)]

    # ------------------------------------------------------------ what moves
    progress(0.45, "finding what moves")
    s = MASK_SIDE / max(h, w)
    mw, mh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    small = [resize_img(im, mw, mh) for im in imgs]
    zsmall = [resize_nearest(d, mw, mh) for d in depth]
    Ksmall = [scale_K(K, mw / w, mh / h) for K in Ks]
    masks_s = moving_masks(small, zsmall, Ksmall, c2w, fps, req.get("mask_offsets_s", (0.5, 1.5, 3.0)),
                           report=lambda i, k: progress(0.45 + 0.04 * i / k, f"finding what moves {i + 1}/{k}"))
    moving = [resize_nearest(m, w, h) for m in masks_s]
    frac = [float(m.mean()) for m in moving]
    log(f"moving pixels: {np.mean(frac):.1%} on average, at most {max(frac):.1%}")
    mask_preview(imgs, moving, os.path.join(out_dir, "moving_preview.png"))

    # second alignment on still pixels only; a fixed camera also shares one background depth
    depth = [aligned(i, trusted[i] & ~moving[i]) for i in range(n)]
    if fixed:
        stack = np.stack([np.where(moving[i] | (depth[i] <= 0), np.nan, depth[i]) for i in range(n)])
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bg_depth = np.nan_to_num(np.nanmedian(stack, 0), nan=0.0).astype(np.float32)
        del stack
        for i in range(n):          # the moving parts in the same depth scale as the shared background
            fit = mv.align_affine(raw[i], bg_depth, (bg_depth > 0) & ~moving[i] & (raw[i] > 0))
            if fit is not None:
                z = raw[i] * fit[0] + fit[1]
                depth[i] = np.where((raw[i] > 0) & (z > 0), z, 0).astype(np.float32)
        priors = [np.where(moving[i], 0, bg_depth).astype(np.float32) for i in range(n)]
    else:
        priors = [np.where(moving[i], 0, depth[i]).astype(np.float32) for i in range(n)]
    raw = zv_all = None          # free memory (no longer needed)

    # ------------------------------------------------------------ background (still part)
    progress(0.5, "building the initial point cloud")
    rng = np.random.default_rng(0)
    use = np.linspace(0, n - 1, min(n, 40)).round().astype(int)
    per_view = max(2000, int(req.get("init_points", 400000) * 1.5 / len(use)))
    pts, cols = [], []
    for i in use:
        yy, xx = np.nonzero(priors[i] > 0)
        if len(yy) > per_view:
            sel = rng.choice(len(yy), per_view, replace=False)
            yy, xx = yy[sel], xx[sel]
        pts.append(unproject(priors[i], Ks[i], c2w[i])[yy, xx])
        cols.append(imgs[i][yy, xx].astype(np.float32) / 255)
    points = np.concatenate(pts).astype(np.float32)
    colors = np.concatenate(cols).astype(np.float32)
    c2ws = np.stack(c2w)
    Ks_arr = np.stack(Ks)
    weights = [float(1 - 0.5 * f) for f in frac]           # frames mostly covered by movers count less
    pixel_w = [(~m).astype(np.float32) for m in moving]
    cfg = st.TrainConfig(steps=int(req.get("steps", 9000)), sh_degree=int(req.get("sh_degree", 1)),
                         max_gaussians=int(req.get("max_gaussians", 1_500_000)),
                         init_points=int(req.get("init_points", 400000)))
    torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.94)))
    train_imgs = imgs
    for attempt in range(3):
        oom = False
        try:
            params, hist = st.train(train_imgs, c2ws, Ks_arr, weights, points, colors, cfg, device="cuda",
                                    progress=lambda v, m: progress(0.5 + 0.36 * v, "background: " + m),
                                    pixel_weights=pixel_w, depth_priors=priors, depth_keep=[True] * n)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or attempt == 2:
                raise
            oom = True
        if oom:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            tw, th = max(8, train_imgs[0].shape[1] * 3 // 4), max(8, train_imgs[0].shape[0] * 3 // 4)
            train_imgs = [resize_img(im, tw, th) for im in train_imgs]
            priors = [resize_nearest(p, tw, th) for p in priors]
            pixel_w = [resize_nearest(p, tw, th) for p in pixel_w]
            Ks_arr = Ks_arr.copy()
            Ks_arr[:, :2] *= 0.75
            cfg.max_gaussians = int(cfg.max_gaussians * 0.6)
            log(f"out of GPU memory while training; retrying at {tw}x{th} with at most {cfg.max_gaussians:,} splats")

    # ------------------------------------------------------------ provenance of the background
    progress(0.87, "labelling the background")
    sizes = [(im.shape[1], im.shape[0]) for im in train_imgs]
    if fixed:            # one viewpoint: depth comes from one camera, it is inferred
        codes = torch.full((len(params["means"]),), INFERRED, dtype=torch.uint8)
    else:
        vm = torch.linalg.inv(torch.as_tensor(c2ws, dtype=torch.float32)).cuda()
        Kt = torch.as_tensor(Ks_arr, dtype=torch.float32).cuda()

        def render_depth(v):
            with torch.no_grad():
                o, alpha, _ = st.gsplat_render(params, vm[v:v + 1], Kt[v:v + 1], sizes[v][0], sizes[v][1], 0,
                                               mode="ED")
            d = o[..., 0]
            return torch.where(alpha[..., 0] > 0.5, d, torch.full_like(d, float("inf")))

        views = [int(k) for k in np.linspace(0, n - 1, min(n, 60)).round()]
        codes = st.provenance_codes(st.visibility_counts(params, c2ws, Ks_arr, sizes, views, render_depth), False)
    with torch.no_grad():
        vm0 = torch.linalg.inv(torch.as_tensor(c2ws[:1], dtype=torch.float32)).cuda()
        rgb, _, _ = st.gsplat_render(params, vm0, torch.as_tensor(Ks_arr[:1], dtype=torch.float32).cuda(),
                                     sizes[0][0], sizes[0][1], cfg.sh_degree)
        still = torch.from_numpy(~resize_nearest(moving[0], *sizes[0])).cuda()
        gt = torch.from_numpy(train_imgs[0]).cuda().float() / 255
        ref_psnr = st.psnr(rgb[..., :3].clamp(0, 1)[still], gt[still])
    bg_path = os.path.join(out_dir, "background.ply")
    count = st.write_ply(params, bg_path, codes)
    log(f"background: {count:,} splats; first-frame PSNR on still pixels {ref_psnr:.2f} dB")
    budget = int(req.get("fg_budget", 40000))

    # ------------------------------------------------------------ moving parts: smooth 4D motion
    dyn, moving_model = None, "per_frame"
    if req.get("motion", True) and any(m.any() for m in moving):
        try:
            dyn = moving_gaussians(params, train_imgs, c2ws, Ks_arr, Ks, imgs, depth, moving, n, fps, budget, req,
                                   torch)
            moving_model = "motion"
        except Exception as e:  # noqa: BLE001 - the per-frame splats below are the fallback
            log(f"smooth 4D motion not used ({type(e).__name__}: {str(e).splitlines()[0][:200]}); "
                "moving parts are rebuilt frame by frame")
            dyn = None
            torch.cuda.empty_cache()
    params = None
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ moving parts: per frame (fallback)
    parts = []
    for i in range(n if dyn is None else 0):
        if i == 0:
            progress(0.9, "moving parts, frame by frame")
        sp = frame_splats(imgs[i], depth[i], moving[i], Ks[i], c2w[i], budget)
        if sp is None:
            continue
        m = len(sp["means"])
        sp["t_center"] = np.full(m, (i + 0.5) / n, np.float32)
        sp["t_scale"] = np.full(m, 0.45 / n, np.float32)
        parts.append(sp)
    keys = ("means", "scales", "quats", "opacity", "colors", "t_center", "t_scale")
    if dyn is None:
        dyn = {k: (np.concatenate([p[k] for p in parts]) if parts else
                   np.zeros((0,) + ((3,) if k in ("means", "scales", "colors") else (4,) if k == "quats" else ()),
                            np.float32)) for k in keys}
        log(f"moving parts: {len(dyn['means']):,} splats over {len(parts)} frames")
    dyn["provenance"] = np.full(len(dyn["means"]), INFERRED, np.uint8)
    dyn_path = os.path.join(out_dir, "dynamic.npz")
    np.savez(dyn_path, **dyn)

    cams = [{"width": w, "height": h, "fx": float(Ks[i][0, 0]), "fy": float(Ks[i][1, 1]), "cx": float(Ks[i][0, 2]),
             "cy": float(Ks[i][1, 2]), "c2w": c2ws[i].tolist(), "file": os.path.basename(paths[i]),
             "generated": False} for i in range(n)]
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cams, f)
    outputs = {"background": bg_path, "dynamic": dyn_path, "cameras": os.path.join(out_dir, "cameras.json"),
               "moving_preview": os.path.join(out_dir, "moving_preview.png")}

    mesh_info = None
    if req.get("mesh", True):
        progress(0.95, "mesh of the background (TSDF fusion)")
        try:
            sel = [int(k) for k in np.linspace(0, n - 1, min(n, 60)).round()]
            dd = [np.where(moving[i], 0, depth[i]).astype(np.float32) for i in sel]
            mesh_info = mv.tsdf_mesh(dd, [Ks[i] for i in sel], [c2ws[i] for i in sel], [imgs[i] for i in sel],
                                     os.path.join(out_dir, "background.obj"), max(med * (scale or 1) / 250, 1e-3))
            outputs["obj"] = os.path.join(out_dir, "background.obj")
        except Exception as e:  # noqa: BLE001 - the mesh is optional
            log(f"mesh export failed: {e}")
    codes_np = codes.cpu().numpy()
    emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), metric=bool(scale),
         metric_scale_factor=scale, frames=n, fps=fps, camera_motion="fixed" if fixed else "moving",
         moving_model=moving_model,
         moving_fraction=[round(x, 4) for x in frac], background_splats=count,
         moving_splats=int(len(dyn["means"])), reference_psnr_db=round(ref_psnr, 2), mesh=mesh_info,
         loss_history=hist[-20:],
         provenance={"observed": int((codes_np == OBSERVED).sum()), "inferred": int((codes_np == INFERRED).sum())
                     + int(len(dyn["means"]))})


if __name__ == "__main__":
    run(main)
