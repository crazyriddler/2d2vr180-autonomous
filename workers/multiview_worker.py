"""Multi-view reconstruction worker (runtime 'recon3d-cu124').

Images (several photos, video keyframes or generated views) →
  VGGT camera poses + depth (chunked, Sim(3)-stitched for long sequences) →
  metric scale from MoGe-2 →
  3D Gaussian Splatting optimisation (splat_trainer.py, gsplat) →
  per-splat provenance (seen by >= 2 real views / 1 real view / generated views only) →
  scene.ply (+ provenance), cameras.json and a TSDF mesh.

Request:
  {"images": [{"path": "...", "generated": false, "weight": 1.0}, ...],   # first = reference view
   "vggt_dir": ".../vggt-1b", "moge_path": ".../model.pt" | null, "output_dir": "...",
   "max_side": 960, "steps": 10000, "max_gaussians": 2000000, "chunk": 24, "overlap": 8,
   "mesh": true, "sh_degree": 3}
Coordinates: OpenCV, scene frame = reference camera.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

VGGT_W = 518
PATCH = 14


def load_rgb(path, max_side=None):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def vggt_input(img):
    """VGGT 'crop' preprocessing with the bookkeeping needed to map results back.

    Returns (array (h,518,3) float32 in [0,1], info) where info maps VGGT pixels to the
    original image: x_orig = x * sx, y_orig = (y + crop) * sy."""
    import numpy as np
    from PIL import Image

    H0, W0 = img.shape[:2]
    new_h = max(PATCH, int(round(H0 * VGGT_W / W0 / PATCH)) * PATCH)
    arr = np.asarray(Image.fromarray(img).resize((VGGT_W, new_h), Image.BICUBIC), np.float32) / 255.0
    crop = 0
    if new_h > VGGT_W:
        crop = (new_h - VGGT_W) // 2
        arr = arr[crop:crop + VGGT_W]
    return arr, {"sx": W0 / VGGT_W, "sy": H0 / new_h, "crop": crop, "h": arr.shape[0]}


def to_original_K(K, info, pad_top):
    import numpy as np

    K = np.array(K, np.float64)
    out = np.eye(3)
    out[0, 0] = K[0, 0] * info["sx"]
    out[1, 1] = K[1, 1] * info["sy"]
    out[0, 2] = K[0, 2] * info["sx"]
    out[1, 2] = (K[1, 2] - pad_top + info["crop"]) * info["sy"]
    return out


def umeyama(src, dst, with_scale=True):
    """Sim(3) (s, R, t) minimising |s R src + t - dst|."""
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if with_scale and var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_sim3(src, dst, iters=3, keep=0.8):
    import numpy as np

    idx = np.arange(len(src))
    s, R, t = umeyama(src, dst)
    for _ in range(iters):
        r = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
        idx = np.argsort(r)[: max(10, int(len(r) * keep))]
        s, R, t = umeyama(src[idx], dst[idx])
    return s, R, t


def free(torch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vggt(model, batch, dtype, torch):
    """batch: list of (h,518,3) arrays → padded tensor → predictions (numpy)."""
    import numpy as np
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    H = max(a.shape[0] for a in batch)
    pads = []
    x = np.ones((len(batch), H, VGGT_W, 3), np.float32)
    for i, a in enumerate(batch):
        top = (H - a.shape[0]) // 2
        x[i, top:top + a.shape[0]] = a
        pads.append(top)
    dev = next(model.parameters()).device
    t = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=dtype):
        pred = model(t)
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], t.shape[-2:])
    out = {"w2c": ext[0].float().cpu().numpy(), "K": intr[0].float().cpu().numpy(),
           "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
           "conf": pred["depth_conf"][0].float().cpu().numpy(), "pads": pads}
    del pred, t
    free(torch)
    return out


def unproject(depth, K, c2w):
    import numpy as np

    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth
    pc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def estimate_cameras(vin, model, dtype, torch, chunk, overlap):
    """Poses (c2w, VGGT pixel frame), intrinsics and depth for every view; the
    first view is the world frame. Long sequences are processed in overlapping
    chunks and stitched with robust Sim(3) fits on shared views."""
    import numpy as np

    n = len(vin)
    c2w = [None] * n
    Ks = [None] * n
    depth = [None] * n
    conf = [None] * n
    starts = [0]
    while starts[-1] + chunk < n:
        starts.append(starts[-1] + chunk - overlap)
    for ci, s0 in enumerate(starts):
        ids = list(range(s0, min(s0 + chunk, n)))
        progress(0.05 + 0.25 * ci / len(starts), f"camera poses: views {ids[0] + 1}-{ids[-1] + 1} of {n}")
        p = run_vggt(model, [vin[i][0] for i in ids], dtype, torch)
        # Work in each image's own (unpadded) VGGT pixel frame: batches pad to different heights.
        loc_d, loc_c, loc_K = [], [], []
        for k, i in enumerate(ids):
            h, pad = vin[i][0].shape[0], p["pads"][k]
            loc_d.append(p["depth"][k][pad:pad + h])
            loc_c.append(p["conf"][k][pad:pad + h])
            Kk = np.array(p["K"][k], np.float64)
            Kk[1, 2] -= pad
            loc_K.append(Kk)
        loc_c2w = [np.linalg.inv(np.vstack([p["w2c"][k], [0, 0, 0, 1]])) for k in range(len(ids))]
        s, R, t = 1.0, np.eye(3), np.zeros(3)
        shared = [k for k, i in enumerate(ids) if c2w[i] is not None]
        if shared:
            src, dst = [], []
            rng = np.random.default_rng(ci)
            for k in shared:
                i = ids[k]
                ok = (loc_c[k] > np.percentile(loc_c[k], 50)) & (conf[i] > np.percentile(conf[i], 50))
                yy, xx = np.nonzero(ok)
                if len(yy) > 4000:
                    sel = rng.choice(len(yy), 4000, replace=False)
                    yy, xx = yy[sel], xx[sel]
                src.append(unproject(loc_d[k], loc_K[k], loc_c2w[k])[yy, xx])
                dst.append(unproject(depth[i], Ks[i], c2w[i])[yy, xx])
            s, R, t = robust_sim3(np.concatenate(src), np.concatenate(dst))
            log(f"chunk {ci}: stitched with scale {s:.4f} on {len(shared)} shared views")
        for k, i in enumerate(ids):
            if c2w[i] is not None:
                continue
            m = np.eye(4)
            m[:3, :3] = R @ loc_c2w[k][:3, :3]
            m[:3, 3] = s * R @ loc_c2w[k][:3, 3] + t
            c2w[i] = m
            Ks[i] = loc_K[k]
            depth[i] = loc_d[k] * s
            conf[i] = loc_c[k]
    # Re-anchor so view 0 is exactly the identity (it is up to numerical noise).
    inv0 = np.linalg.inv(c2w[0])
    c2w = [inv0 @ m for m in c2w]
    return c2w, Ks, depth, conf


def metric_scale(moge_path, vin, depth, conf, torch, max_views=4, device="cuda"):
    import numpy as np
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(moge_path).to(device).eval()
    ratios = []
    for i in np.linspace(0, len(vin) - 1, min(max_views, len(vin))).round().astype(int):
        a = vin[i][0]
        t = torch.from_numpy(a).to(device).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(t, use_fp16=device == "cuda")
        mz = out["depth"].float().cpu().numpy()
        vz, vc = depth[i], conf[i]
        ok = np.isfinite(mz) & (mz > 0) & (vz > 0) & (vc > np.percentile(vc, 50))
        if "mask" in out:
            ok &= out["mask"].cpu().numpy().astype(bool)
        if ok.sum() > 100:
            ratios.append(float(np.median(mz[ok] / vz[ok])))
    del model
    free(torch)
    return float(np.median(ratios)) if ratios else None


def tsdf_mesh(depths, Ks, c2ws, colors, out_obj, voxel):
    import numpy as np
    import open3d as o3d

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 5, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for d, K, c2w, col in zip(depths, Ks, c2ws, colors):
        if not np.any(d > 0):
            continue
        h, w = d.shape
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(col)), o3d.geometry.Image(np.ascontiguousarray(d, np.float32)),
            depth_scale=1.0, depth_trunc=float(np.percentile(d[d > 0], 98)) * 1.001, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(c2w))
    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    # OBJ viewers expect y up: flip OpenCV (y down, z forward)
    mesh.transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    o3d.io.write_triangle_mesh(out_obj, mesh, write_vertex_colors=True)
    return {"vertices": len(mesh.vertices), "faces": len(mesh.triangles), "voxel_m": voxel}


def vggt_on_image(map_v, info, h, w):
    """Sample a VGGT-resolution map (cropped VGGT frame, see vggt_input) on an (h, w) image of the same view.
    Returns (map at image pixels, mask of pixels that VGGT saw)."""
    import numpy as np

    r = w / (info["sx"] * info.get("w", VGGT_W))
    xs = ((np.arange(w) + 0.5) / r / info["sx"]).astype(int)
    ys = ((np.arange(h) + 0.5) / r / info["sy"] - info["crop"]).astype(int)
    inside_y = (ys >= 0) & (ys < map_v.shape[0])
    xs = np.clip(xs, 0, map_v.shape[1] - 1)
    yc = np.clip(ys, 0, map_v.shape[0] - 1)
    return map_v[yc][:, xs], np.broadcast_to(inside_y[:, None], (h, w))


def consistency_map(ref_img, ref_depth, K0, c2w0, img, depth, K, c2w, cell=4, sigma=0.12, floor=0.1):
    """Per-pixel confidence (H, W) in [floor, 1] for a generated view: the photo is reprojected into it
    (with the photo's depth) and colours are compared where both see the same surface, after removing
    the overall colour/brightness offset between the two images. Pixels the photo does not see (new
    content) get 1 - they are what the view is for."""
    import numpy as np

    h, w = depth.shape
    X = unproject(ref_depth, K0, c2w0)
    ok0 = np.isfinite(ref_depth) & (ref_depth > 0)
    X, cols = X[ok0], ref_img[ok0].astype(np.float32) / 255.0
    w2c = np.linalg.inv(c2w)
    pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K[0, 0] * pc[:, 0] / z + K[0, 2])
        v = np.floor(K[1, 1] * pc[:, 1] / z + K[1, 2])
    inb = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui, vi = u[inb].astype(np.int64), v[inb].astype(np.int64)
    zj = depth[vi, ui]
    vis = (zj > 0) & (z[inb] <= zj * 1.08)
    ui, vi, c0 = ui[vis], vi[vis], cols[inb][vis]
    c1 = img[vi, ui].astype(np.float32) / 255.0
    if len(c0) < 200:
        return np.ones((h, w), np.float32)
    c1 = c1 - np.median(c1 - c0, axis=0)            # global colour / exposure difference
    diff = np.abs(c1 - c0).mean(1)
    gh, gw = (h + cell - 1) // cell, (w + cell - 1) // cell
    acc = np.zeros(gh * gw, np.float64)
    cnt = np.zeros(gh * gw, np.float64)
    idx = (vi // cell) * gw + (ui // cell)
    np.add.at(acc, idx, diff)
    np.add.at(cnt, idx, 1.0)
    conf = np.ones(gh * gw, np.float32)
    seen = cnt > 0
    conf[seen] = np.exp(-((acc[seen] / cnt[seen]) / sigma) ** 2)
    conf = np.maximum(conf.reshape(gh, gw), floor)
    return np.repeat(np.repeat(conf, cell, 0), cell, 1)[:h, :w].astype(np.float32)


def subject_mask(depth, valid=None, min_gap=1.25, min_sep=0.85, min_frac=0.03, max_frac=0.9):
    """Subject (foreground) mask of a view from its monocular depth: Otsu threshold on log depth.

    Returns (mask or None, reason). None when the depth is not clearly two-layered (a landscape, a
    room): then there is no single subject in front of a background and the whole image is used."""
    import numpy as np
    from scipy import ndimage

    d = np.asarray(depth, np.float64)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    if ok.mean() < 0.3:
        return None, "too little valid depth"
    ld = np.log(d[ok])
    if ld.var() < 1e-8:
        return None, "flat depth"
    hist, edges = np.histogram(ld, 256)
    p = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (m0[-1] * w0 - m0) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = 0
    k = int(np.argmax(sb))
    t = c[k]
    # share of the depth variance explained by two layers: ~0.99 for a person in front of a wall, 0.90 with
    # a floor, 0.75 for a continuous ramp of depths (a landscape, a room) - which has no single subject
    sep = float(sb[k] / ld.var())
    near, far = ld[ld <= t], ld[ld > t]
    if len(near) == 0 or len(far) == 0:
        return None, "one depth layer"
    gap = float(np.exp(far.mean() - near.mean()))
    if sep < min_sep or gap < min_gap:
        return None, f"no separate subject (layer separation {sep:.2f}, depth ratio {gap:.2f})"
    fg = ok & (np.log(np.where(ok, d, 1.0)) <= t)
    lab, nlab = ndimage.label(fg)
    if nlab > 1:
        sizes = ndimage.sum(fg, lab, range(1, nlab + 1))
        fg = np.isin(lab, 1 + np.nonzero(sizes >= 0.2 * sizes.max())[0])
    fg = ndimage.binary_fill_holes(fg)
    fg = ndimage.binary_dilation(fg, iterations=max(1, round(0.004 * max(fg.shape))))
    frac = float(fg.mean())
    if not min_frac <= frac <= max_frac:
        return None, f"subject covers {frac:.0%} of the image"
    return fg, f"subject {frac:.0%} of the image (layer separation {sep:.2f}, depth ratio {gap:.2f})"


def subject_depths(moge_path, paths, max_side, device, torch):
    """(MoGe-2 depth, valid mask) for every image path, at a moderate resolution."""
    import numpy as np
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(moge_path).to(device).eval()
    out = {}
    for k, path in enumerate(paths):
        progress(0.02 + 0.01 * k / max(1, len(paths)), f"finding the subject {k + 1}/{len(paths)}")
        img = load_rgb(path, max_side)
        t = torch.from_numpy(np.ascontiguousarray(img)).to(device).float().div(255).permute(2, 0, 1)
        with torch.no_grad():
            res = model.infer(t, resolution_level=6, use_fp16=device == "cuda")
        valid = res["mask"].cpu().numpy().astype(bool) if "mask" in res else None
        out[path] = (res["depth"].float().cpu().numpy(), valid)
        del res, t
    del model
    free(torch)
    return out


def masked(img, mask, grey=128):
    """The image with everything outside the subject mask replaced by a flat grey."""
    import numpy as np

    if mask is None:
        return img
    m = resize_map(mask.astype(np.uint8), img.shape[1], img.shape[0]).astype(bool)
    return np.where(m[..., None], img, np.uint8(grey)).astype(np.uint8)


def mono_priors(moge_path, imgs, vin, depth, conf, device, torch, align_masks=None):
    """MoGe-2 depth of every view at its training resolution, scaled onto the view's VGGT depth.
    Returns a list of (H, W) depth maps in scene units (0 where unknown)."""
    import numpy as np
    from moge.model.v2 import MoGeModel

    import fusion as fu

    model = MoGeModel.from_pretrained(moge_path).to(device).eval()
    out = []
    for i, img in enumerate(imgs):
        progress(0.33 + 0.02 * i / len(imgs), f"sharp depth priors (MoGe-2) {i + 1}/{len(imgs)}")
        h, w = img.shape[:2]
        t = torch.from_numpy(img).to(device).float().div(255).permute(2, 0, 1)
        with torch.no_grad():
            res = model.infer(t, resolution_level=9, use_fp16=device == "cuda")
        zm = res["depth"].float().cpu().numpy()
        valid = np.isfinite(zm) & (zm > 0)
        if "mask" in res:
            valid &= res["mask"].cpu().numpy().astype(bool)
        del res, t
        zv, inside = vggt_on_image(depth[i], vin[i][1], h, w)
        cv, _ = vggt_on_image(conf[i], vin[i][1], h, w)
        sel = valid & inside & (cv >= np.percentile(conf[i], 50))
        if align_masks is not None and align_masks[i] is not None:
            sel &= resize_map(align_masks[i].astype(np.uint8), w, h).astype(bool)   # the subject only
        s = fu.align_scale(zm, zv, sel)
        out.append(np.where(valid, zm * s, 0.0).astype(np.float32) if s else None)
    del model
    free(torch)
    return out


def resize_map(m, w, h):
    """Nearest-neighbour resize of a 2-D map (None passes through)."""
    if m is None:
        return None
    import fusion as fu

    return fu.resample(m, (h, w))


def fuse_views(req, items, vin, c2w_v, K_v, depth, conf, device, torch, out_dir):
    """Sharp assembly (generated views): MoGe-2 depth per view, scaled onto the VGGT depth of that
    view, merged by coverage (fusion.py) - no optimisation, so nothing is averaged into blur."""
    import numpy as np
    from moge.model.v2 import MoGeModel

    import fusion as fu
    import splat_trainer as st

    model = MoGeModel.from_pretrained(req["moge_path"]).to(device).eval()
    ref_side, side = int(req.get("fuse_ref_side", 1536)), int(req.get("fuse_side", 1024))
    views, scales = [], []
    for i, it in enumerate(items):
        progress(0.36 + 0.4 * i / len(items), f"sharp geometry (MoGe-2) {i + 1}/{len(items)}")
        img = load_rgb(it["path"], ref_side if i == 0 else side)
        h, w = img.shape[:2]
        t = torch.from_numpy(img).to(device).float().div(255).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(t, resolution_level=9, use_fp16=device == "cuda")
        zm = out["depth"].float().cpu().numpy()
        valid = np.isfinite(zm) & (zm > 0)
        if "mask" in out:
            valid &= out["mask"].cpu().numpy().astype(bool)
        del out, t
        # this image's pixels → the view's VGGT pixels (see vggt_input)
        info = vin[i][1]
        r = w / (info["sx"] * info.get("w", VGGT_W))
        zv, inside = vggt_on_image(depth[i], info, h, w)
        cv, _ = vggt_on_image(conf[i], info, h, w)
        ok = valid & inside & (cv > np.percentile(conf[i], 50))
        s = fu.align_scale(zm, zv, ok)
        if s is None:
            if i == 0:
                s = 1.0
            else:
                log(f"view {i}: could not align its depth; skipped")
                continue
        scales.append(s)
        K = to_original_K(K_v[i], info, 0)
        K[:2] *= r
        views.append(fu.View(zm * s, K, c2w_v[i], img, valid, bool(it.get("generated"))))
        if i == 0:
            ref_index = len(views) - 1
    del model
    free(torch)
    log("depth alignment scales: " + ", ".join(f"{s:.3f}" for s in scales))
    progress(0.8, f"fusing {len(views)} views")
    order = fu.fusion_order([v.c2w for v in views], ref_index)
    fu.fuse(views, order)
    for k in order:
        v = views[k]
        log(f"view {k}: {int(v.keep.sum()):,} of {int(v.valid.sum()):,} pixels kept"
            + (" (generated)" if v.generated else " (photo)"))
    arrs, prov = fu.assemble(views, int(req.get("max_gaussians", 3_000_000)))
    n = len(prov)
    params = {k: torch.from_numpy(a) for k, a in arrs.items()}
    params["shN"] = torch.zeros(n, 0, 3)
    ply = os.path.join(out_dir, "scene.ply")
    count = st.write_ply(params, ply, prov)
    log(f"{count:,} splats")
    cams = [{"width": v.w, "height": v.h, "fx": float(v.K[0, 0]), "fy": float(v.K[1, 1]), "cx": float(v.K[0, 2]),
             "cy": float(v.K[1, 2]), "c2w": v.c2w.tolist(), "file": os.path.basename(items[0]["path"]) if j == ref_index
             else f"view{j}", "generated": v.generated} for j, v in enumerate(views)]
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cams, f)
    outputs = {"ply": ply, "cameras": os.path.join(out_dir, "cameras.json")}
    mesh_info = None
    if req.get("mesh", True):
        progress(0.9, "mesh (TSDF fusion)")
        try:
            dd = [np.nan_to_num(v.z, nan=0.0) for v in views]
            med = float(np.median(dd[ref_index][dd[ref_index] > 0]))
            mesh_info = tsdf_mesh(dd, [v.K for v in views], [v.c2w for v in views], [v.image for v in views],
                                  os.path.join(out_dir, "scene.obj"), max(med / 250, 1e-3))
            outputs["obj"] = os.path.join(out_dir, "scene.obj")
        except Exception as e:  # noqa: BLE001 - the mesh is optional
            log(f"mesh export failed: {e}")
    return outputs, count, prov, mesh_info


# ----------------------------------------------------------------------------- Depth Anything 3
def da3_stub():
    """DA3's API module imports its exporters (pycolmap, trimesh, moviepy, gsplat) and pose alignment
    (evo) at import time; inference needs none of them, so those two modules are replaced."""
    import types

    def nope(*a, **k):
        raise RuntimeError("this Depth Anything 3 feature is not bundled with 2D2VR180")

    def stub_attr(attr):           # any function the model files import from them (e.g. the Giant
        if attr.startswith("__"):  # models' Gaussian head imports batch_align_poses_umeyama)
            raise AttributeError(attr)
        return nope

    for name in ("depth_anything_3.utils.export", "depth_anything_3.utils.pose_align"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__getattr__ = stub_attr
            sys.modules[name] = m


def load_da3(model_dir, device):
    da3_stub()
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(model_dir).to(device).eval()


def try_load_da3(model_dir, device, torch):
    """Depth Anything 3, or None (logged) when it cannot be loaded: the caller then uses VGGT."""
    try:
        return load_da3(model_dir, device)
    except Exception as e:  # noqa: BLE001 - VGGT is the fallback engine
        log(f"Depth Anything 3 could not be loaded ({type(e).__name__}: {e}); using VGGT instead")
        free(torch)
        return None


def da3_size(img, res):
    """Common processing size (h, w): longest side `res`, both multiples of DA3's 14-pixel patch."""
    H0, W0 = img.shape[:2]
    s = res / max(H0, W0)
    return max(14, int(round(H0 * s / 14)) * 14), max(14, int(round(W0 * s / 14)) * 14)


def da3_input(img, size, mask=None):
    """(float image at the DA3 size, info) - info maps DA3 pixels to the image like vggt_input's.
    With a subject mask the background is flattened to grey (the cameras are then found from the
    subject alone) and the mask at the DA3 size is kept in info["mask"]."""
    import numpy as np
    from PIL import Image

    h, w = size
    H0, W0 = img.shape[:2]
    arr = np.asarray(Image.fromarray(masked(img, mask)).resize((w, h), Image.BICUBIC), np.float32) / 255.0
    info = {"sx": W0 / w, "sy": H0 / h, "crop": 0, "h": h, "w": w}
    if mask is not None:
        info["mask"] = resize_map(mask.astype(np.uint8), w, h).astype(bool)
    return arr, info


def quat_from_rotmat(R):
    """Unit quaternion (w, x, y, z) of a 3x3 rotation matrix."""
    import numpy as np

    R = np.asarray(R, np.float64)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Hamilton product of (w, x, y, z) quaternions; a is (4,), b is (N, 4)."""
    import numpy as np

    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def gaussians_to_frame(means, scales, quats, scale_factor, inv0):
    """Depth Anything 3's feed-forward Gaussians are built in the model's frame before its metric
    scaling: scale them like its depth and cameras (x scale_factor), then re-anchor them like the
    cameras (photo camera = identity, inv0 = inverse of the photo's predicted camera-to-world)."""
    import numpy as np

    s = float(scale_factor) if scale_factor else 1.0
    R, t = np.asarray(inv0)[:3, :3], np.asarray(inv0)[:3, 3]
    means = (np.asarray(means, np.float64) * s) @ R.T + t
    quats = quat_mul(quat_from_rotmat(R), np.asarray(quats, np.float64))
    return means.astype(np.float32), (np.asarray(scales, np.float64) * s).astype(np.float32), \
        (quats / np.linalg.norm(quats, axis=1, keepdims=True)).astype(np.float32)


def da3_dc_only(sh, rotations):
    """Stand-in for DA3's rotate_sh (needs e3nn): keeps the view-independent colour (DC band, which
    rotation does not change) and drops the higher bands."""
    out = sh.clone()
    out[..., 1:] = 0
    return out


def da3_predict(model, arrays, torch, infer_gs=False):
    """Poses, intrinsics, depth and confidence for same-size float images; first view = world frame.
    Same keys as run_vggt (pads are 0: no padding)."""
    import numpy as np

    imgs = [(np.clip(a, 0, 1) * 255).round().astype(np.uint8) for a in arrays]
    h, w = imgs[0].shape[:2]
    if infer_gs:
        import depth_anything_3.model.gs_adapter as ga

        ga.rotate_sh = da3_dc_only
    with torch.no_grad():
        pred = model.inference(imgs, process_res=max(h, w), process_res_method="upper_bound_resize",
                               ref_view_strategy="first", infer_gs=infer_gs)
    if tuple(pred.depth.shape[1:]) != (h, w):
        raise RuntimeError(f"Depth Anything 3 changed the image size {(h, w)} -> {tuple(pred.depth.shape[1:])}")
    c2w = [np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in np.asarray(pred.extrinsics, np.float64)]
    inv0 = np.linalg.inv(c2w[0])
    w2c = [np.linalg.inv(inv0 @ m)[:3] for m in c2w]
    out = {"w2c": np.stack(w2c), "K": np.asarray(pred.intrinsics, np.float64),
           "depth": np.asarray(pred.depth, np.float32),
           "conf": (np.asarray(pred.conf, np.float32) if pred.conf is not None
                    else np.ones(np.asarray(pred.depth).shape, np.float32)),
           "pads": [0] * len(imgs)}
    if infer_gs and getattr(pred, "gaussians", None) is not None:
        g = pred.gaussians
        V = len(imgs)
        means, scales, quats = gaussians_to_frame(
            g.means[0].float().cpu().numpy(), g.scales[0].float().cpu().numpy(),
            g.rotations[0].float().cpu().numpy(), getattr(pred, "scale_factor", None), inv0)
        op = g.opacities[0].float().cpu().numpy()
        out["gaussians"] = {"means": means.reshape(V, h, w, 3), "scales": scales.reshape(V, h, w, 3),
                            "quats": quats.reshape(V, h, w, 4),
                            "dc": g.harmonics[0][..., 0].float().cpu().numpy().reshape(V, h, w, 3),
                            "opacity": op.reshape(V, h, w, -1)[..., 0]}
        del g
    free(torch)
    return out


def da3_cameras(model, vin, torch, infer_gs=False):
    """estimate_cameras() with Depth Anything 3 (all views at once). With infer_gs, also its
    feed-forward Gaussians (per pixel of every view, in the same frame) as a 5th value."""
    import numpy as np

    progress(0.1, f"camera poses and depth (Depth Anything 3, {len(vin)} views)")
    p = da3_predict(model, [v[0] for v in vin], torch, infer_gs=infer_gs)
    c2w = [np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in p["w2c"]]
    res = (c2w, [np.array(k) for k in p["K"]], list(p["depth"]), list(p["conf"]))
    return res + (p.get("gaussians"),) if infer_gs else res


def rigidity_score(model, ref_vin, cand_vin, dtype, torch, target_deg=None, predict=None):
    """How well a generated view is explained as a pure camera move of the reference photo.

    VGGT poses the pair and gives the photo's depth; the photo is reprojected into the candidate's
    view and compared where both see the same surface. A changed pose, head turn or expression
    shows up as colour mismatch. Lower is better. Returns (score, colour error, angle°, overlap)."""
    import math

    import numpy as np

    p = (predict or (lambda arrays: run_vggt(model, arrays, dtype, torch)))([ref_vin[0], cand_vin[0]])
    h0, h1 = ref_vin[0].shape[0], cand_vin[0].shape[0]
    pad0, pad1 = p["pads"]
    d0, c0 = p["depth"][0][pad0:pad0 + h0], p["conf"][0][pad0:pad0 + h0]
    d1 = p["depth"][1][pad1:pad1 + h1]
    K0, K1 = np.array(p["K"][0], np.float64), np.array(p["K"][1], np.float64)
    K0[1, 2] -= pad0
    K1[1, 2] -= pad1
    c2w0 = np.linalg.inv(np.vstack([p["w2c"][0], [0, 0, 0, 1]]))
    w2c1 = np.vstack([p["w2c"][1], [0, 0, 0, 1]])
    X = unproject(d0, K0, c2w0).reshape(-1, 3)
    pc = X @ w2c1[:3, :3].T + w2c1[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K1[0, 0] * pc[:, 0] / z + K1[0, 2])
        v = np.floor(K1[1, 1] * pc[:, 1] / z + K1[1, 2])
    w1 = cand_vin[0].shape[1]
    inb = (z > 1e-6) & (u >= 0) & (u < w1) & (v >= 0) & (v < h1)
    ui = np.where(inb, u, 0).astype(np.int64)
    vi = np.where(inb, v, 0).astype(np.int64)
    ok = inb & (z <= d1[vi, ui] * 1.05) & (c0.reshape(-1) >= np.percentile(c0, 50))
    if ref_vin[1].get("mask") is not None:                 # subject mode: the grey background says nothing
        ok &= ref_vin[1]["mask"].reshape(-1)
        if cand_vin[1].get("mask") is not None:
            ok &= cand_vin[1]["mask"][vi, ui]
    rel = w2c1 @ c2w0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel[:3, :3]) - 1) / 2))))
    overlap = float(ok.mean())
    if ok.sum() < 500:
        return 10.0, None, angle, overlap
    a = ref_vin[0].reshape(-1, 3)[ok]
    b = cand_vin[0][vi[ok], ui[ok]]
    b = b - np.median(b - a, axis=0)          # overall exposure / white-balance difference is not a change
    diff = np.abs(a - b).mean(1)
    # Changes are local (a turned head, a moved hand): the share of clearly mismatching pixels catches
    # them, which a median would ignore; the mean error adds the overall fit.
    err = float((diff > 0.12).mean() + diff.mean())
    score = err + 0.5 * max(0.0, 0.25 - overlap)          # little shared surface: less trustworthy
    if target_deg:
        # it must have moved by about the angle asked for: a "30°" view measured at 79° is not a clean
        # camera move (rc31 test), nor a "45°" one measured at 6° (rc30)
        score += 0.01 * max(0.0, abs(angle - target_deg) - 15)
    return score, err, angle, overlap


def select_candidates(items, train_imgs, vin, model, dtype, torch, max_side, prep=None, predict=None,
                      min_candidates=2):
    """For every view with several generated candidates keep the most rigid one (rigidity_score)."""
    report = []
    for i, it in enumerate(items):
        cands = it.get("candidates") or []
        if len(cands) < min_candidates:
            continue
        scored = []
        for path in cands:
            img = load_rgb(path, max_side)
            cv = prep(img, path) if prep else vggt_input(img)
            sc, err, ang, ov = rigidity_score(model, vin[0], cv, dtype, torch, it.get("target_deg"), predict)
            scored.append((sc, path, img, cv, err, ang, ov))
            log(f"view {i} ({it.get('label', '')}) candidate {os.path.basename(path)}: score {sc:.4f} "
                f"(colour error {err if err is None else round(err, 4)}, camera moved {ang:.1f}°, "
                f"shared surface {ov:.0%})")
        best = min(scored, key=lambda t: t[0])
        items[i] = dict(it, path=best[1], rigidity=best[0])
        train_imgs[i], vin[i] = best[2], best[3]
        report.append({"view": it.get("label", str(i)), "chosen": os.path.basename(best[1]),
                       "scores": {os.path.basename(t[1]): round(t[0], 4) for t in scored},
                       "camera_moved_deg": round(best[5], 1)})
        log(f"view {i} ({it.get('label', '')}): kept {os.path.basename(best[1])}")
    return report


def turntable_c2ws(target_z, n=72, yaw_deg=30.0, pitch_deg=6.0):
    """Camera path that swings around the subject: starts at the photo's camera, goes yaw_deg to each
    side (with a slight rise and fall) and comes back. OpenCV frame, photo camera = identity."""
    import math

    import numpy as np

    c = np.array([0.0, 0.0, float(target_z)])
    out = []
    for k in range(n):
        t = k / n
        yaw = math.radians(yaw_deg) * math.sin(2 * math.pi * t)
        pitch = math.radians(pitch_deg) * math.sin(4 * math.pi * t)
        cy, sy, cp, sp = math.cos(-yaw), math.sin(-yaw), math.cos(pitch), math.sin(pitch)
        R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = c - R @ c
        out.append(m)
    return out


def turntable_orbit(means, c2ws, K, size, mask=None, max_yaw=30.0, min_yaw=8.0):
    """(orbit distance, swing in degrees) for the turntable preview.

    The orbit centre is the median depth of the splats that the photo shows inside the subject mask
    (all splats in front of the camera without one), so the camera swings around the person, not
    around the wall behind. The swing stays within ~70 % of the widest angle the views cover, so the
    preview shows what was reconstructed rather than what nobody saw."""
    import math

    import numpy as np

    c2ws = np.asarray(c2ws, np.float64)
    w2c0 = np.linalg.inv(c2ws[0])
    pc = means @ w2c0[:3, :3].T + w2c0[:3, 3]
    z = pc[:, 2]
    front = z > 1e-6
    if mask is not None and front.any():
        w, h = size
        m = resize_map(np.asarray(mask, np.uint8), w, h).astype(bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(K[0, 0] * pc[:, 0] / z + K[0, 2])
            v = np.floor(K[1, 1] * pc[:, 1] / z + K[1, 2])
        inb = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        sel = np.zeros(len(z), bool)
        sel[inb] = m[v[inb].astype(int), u[inb].astype(int)]
        if sel.sum() > 100:
            front = sel
    zc = float(np.median(z[front])) if front.any() else 2.0
    widest = 0.0
    for c in c2ws[1:]:
        rel = w2c0 @ c
        widest = max(widest, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel[:3, :3]) - 1) / 2)))))
    yaw = max(min_yaw, min(max_yaw, 0.7 * widest)) if len(c2ws) > 1 else min_yaw
    return zc, yaw


def render_turntable(params, K, size, target_z, out_dir, sh_degree, torch, n=72, width=960, base=None,
                     yaw_deg=30.0):
    """JPEG frames of the trained splat seen along turntable_c2ws (the app encodes them to MP4)."""
    import numpy as np
    from PIL import Image

    import splat_trainer as st

    w0, h0 = size
    s = min(1.0, width / w0)
    w, h = int(round(w0 * s / 2)) * 2, int(round(h0 * s / 2)) * 2
    Kr = np.array(K, np.float64).copy()
    Kr[:2] *= [[w / w0], [h / h0]]
    Kt = torch.as_tensor(Kr, dtype=torch.float32, device=params["means"].device)[None]
    os.makedirs(out_dir, exist_ok=True)
    base = np.eye(4) if base is None else np.asarray(base, np.float64)
    for k, c2w in enumerate(turntable_c2ws(target_z, n, yaw_deg=yaw_deg)):
        c2w = base @ c2w
        vm = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32, device=params["means"].device))[None]
        with torch.no_grad():
            rgb, _, _ = st.gsplat_render(params, vm, Kt, w, h, sh_degree)
        img = (rgb[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
        Image.fromarray(img).save(os.path.join(out_dir, f"frame_{k:03d}.jpg"), quality=90)
    return n


def splat_depth_views(params, c2ws, Ks, sizes, torch, max_views=60, max_side=1024, min_alpha=0.6):
    """Depth (expected depth, 0 where the splat is transparent) and colour of the
    trained splat seen from the training cameras, for TSDF meshing."""
    import numpy as np

    import splat_trainer as st

    dev = params["means"].device
    idx = list(range(len(c2ws)))[:: max(1, len(c2ws) // max_views)]
    dd, cc, ks, cs = [], [], [], []
    for i in idx:
        w0, h0 = sizes[i]
        s = min(1.0, max_side / max(w0, h0))
        w, h = max(8, int(w0 * s)), max(8, int(h0 * s))
        K = np.array(Ks[i], np.float64).copy()
        K[:2] *= [[w / w0], [h / h0]]
        vm = torch.linalg.inv(torch.as_tensor(np.asarray(c2ws[i]), dtype=torch.float32, device=dev))[None]
        Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            out, alpha, _ = st.gsplat_render(params, vm, Kt, w, h, 0, mode="RGB+ED")
        a = alpha[..., 0]
        d = torch.where(a > min_alpha, out[..., 3], torch.zeros_like(a))     # "ED" is already / alpha
        dd.append(d.float().cpu().numpy())
        cc.append((out[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy())
        ks.append(K)
        cs.append(np.asarray(c2ws[i], np.float64))
    return dd, cc, ks, cs


def ff_init_gaussians(G, items, vin, depth, subj, photo_depth, photo_img, K0, c2w0, far_pct=90.0):
    """Initial Gaussians from Depth Anything 3's feed-forward splat (per pixel of every view):
    border pixels and the farthest 10 % are left out (as DA3's own exporter does), generated views
    keep only their subject in subject mode, and there the photo's background is added from its
    own depth (photo_depth, at the photo's training resolution)."""
    import numpy as np

    V, h, w = G["means"].shape[:3]
    th, tw = max(1, int(8 / 256 * h)), max(1, int(8 / 256 * w))
    out = {k: [] for k in G}
    for v in range(V):
        m = np.zeros((h, w), bool)
        m[th:-th, tw:-tw] = True
        if subj[v] is not None and vin[v][1].get("mask") is not None:
            m &= vin[v][1]["mask"]
        else:
            d = depth[v]
            m &= d <= np.percentile(d[d > 0], far_pct) if np.any(d > 0) else True
        m &= np.isfinite(G["means"][v]).all(-1)
        for k in G:
            out[k].append(G[k][v][m])
    g = {k: np.concatenate(a) for k, a in out.items()}
    n_ff = len(g["means"])
    if subj[0] is not None and photo_depth is not None:
        H, W = photo_depth.shape
        stride = max(1, int(round(W / (1.4 * w))))
        bg = ~resize_map(subj[0].astype(np.uint8), W, H).astype(bool) & (photo_depth > 0)
        sub = np.zeros_like(bg)
        sub[::stride, ::stride] = True
        yy, xx = np.nonzero(bg & sub)
        if len(yy):
            X = unproject(photo_depth, K0, c2w0)[yy, xx]
            sc = (photo_depth[yy, xx] * stride / K0[0, 0] * 0.8).astype(np.float32)
            q = np.zeros((len(yy), 4), np.float32)
            q[:, 0] = 1
            col = photo_img[yy, xx].astype(np.float32) / 255.0
            g = {"means": np.concatenate([g["means"], X.astype(np.float32)]),
                 "scales": np.concatenate([g["scales"], np.repeat(sc[:, None], 3, 1)]),
                 "quats": np.concatenate([g["quats"], q]),
                 "dc": np.concatenate([g["dc"], (col - 0.5) / 0.28209479177387814]),
                 "opacity": np.concatenate([g["opacity"], np.full(len(yy), 0.9, np.float32)])}
    log(f"feed-forward splat: {n_ff:,} Gaussians from the views"
        + (f" + {len(g['means']) - n_ff:,} for the photo's background" if len(g["means"]) > n_ff else ""))
    return g


def drop_inconsistent(items, train_imgs, vin, threshold):
    """Remove generated views whose best candidate is still far from a pure camera move (they would
    teach the splat a different pose or face), keeping at least one generated view."""
    bad = [i for i, it in enumerate(items) if it.get("generated") and it.get("rigidity", 0) > threshold]
    gen = [i for i, it in enumerate(items) if it.get("generated")]
    if bad and len(bad) >= len(gen):
        bad = sorted(bad, key=lambda i: items[i]["rigidity"])[1:]    # keep the least bad one
    labels = [items[i].get("label", str(i)) for i in bad]
    for i in sorted(bad, reverse=True):
        log(f"view {i} ({items[i].get('label', '')}): dropped - even its best candidate does not match the photo "
            f"(score {items[i]['rigidity']:.3f} > {threshold})")
        del items[i], train_imgs[i], vin[i]
    return labels


def vggt_stage(req, items, train_imgs, vin, device, torch, max_side, env):
    """VGGT poses + depth (chunked for long sequences), with candidate selection first."""
    from vggt.models.vggt import VGGT

    progress(0.03, "loading VGGT")
    vdir = req["vggt_dir"]
    if os.path.exists(os.path.join(vdir, "config.json")) and os.path.exists(os.path.join(vdir, "model.safetensors")):
        model = VGGT.from_pretrained(vdir)
    else:
        model = VGGT()
        sd = torch.load(os.path.join(vdir, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
    model = model.to(device).eval()
    dtype = (torch.bfloat16 if device == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    vram = env.get("vram_total_mib", 16000) / 1024
    chunk = int(req.get("chunk") or (24 if vram >= 15 else 12))
    selection = []
    if any(len(it.get("candidates") or []) > 1 for it in items):
        progress(0.04, "choosing the most consistent generated views")
        selection = select_candidates(items, train_imgs, vin, model, dtype, torch, max_side)
        dropped = drop_inconsistent(items, train_imgs, vin, float(req.get("drop_threshold", 0.8)))
        for sel in selection:
            sel["dropped"] = sel["view"] in dropped
    c2w_v, K_v, depth, conf = estimate_cameras(vin, model, dtype, torch, chunk, int(req.get("overlap", 8)))
    del model
    free(torch)
    return c2w_v, K_v, depth, conf, vin, selection


def subject_setup(req, items, max_side, device, torch):
    """Subject mode for photo + generated views: {path: mask or None} for the photo, every generated
    view and its candidates, or None to use whole images.

    Image models redraw a person from a new angle but keep a plain studio backdrop as it was, so
    the background says "the camera barely moved" while the person says "45°". Finding the cameras
    from the subject alone, and letting generated views teach only the subject, avoids a 3D torn
    between the two (the photo alone provides the background)."""
    if (str(req.get("subject_mode", "auto")) == "off" or not req.get("moge_path")
            or req.get("assembly") == "fusion" or not any(it.get("generated") for it in items)):
        return None
    paths = []
    for it in items:
        for path in [it["path"]] + list(it.get("candidates") or []):
            if path not in paths:
                paths.append(path)
    try:
        res = subject_depths(req["moge_path"], paths, max_side, device, torch)
    except Exception as e:  # noqa: BLE001 - subject mode is an improvement, not a requirement
        log(f"subject detection unavailable ({e}); using whole images")
        return None
    m0, why = subject_mask(*res[items[0]["path"]])
    log(f"photo: {why}")
    if m0 is None:
        log("no single subject in front of a background: cameras are found from the whole images")
        return None
    # the photo has shown a subject in front of a background: in the generated views (the same scene)
    # take their near layer without asking for as clean a separation
    masks = {items[0]["path"]: m0}
    for path in paths[1:]:
        m, why_v = subject_mask(*res[path], min_gap=1.15, min_sep=0.6, min_frac=0.02, max_frac=0.95)
        masks[path] = m
        log(f"{os.path.basename(path)}: {why_v}")
    missing = [path for path, m in masks.items() if m is None]
    if missing:
        log(f"no subject found in {', '.join(os.path.basename(m) for m in missing)}: those candidates are "
            "not used (a whole image next to subject-only views would mislead the camera engine)")
    log("subject mode: cameras from the subject; generated views teach only the subject, the background "
        "comes from the photo")
    return masks


def keep_masked_views(items, train_imgs, masks, max_side):
    """Subject mode: generated views and candidates without a subject mask are left out (in place).
    Returns the masks, or None when no generated view is left (then whole images are used)."""
    keep = []
    for i, it in enumerate(items):
        if not it.get("generated"):
            keep.append(i)
            continue
        cands = list(dict.fromkeys([it["path"]] + list(it.get("candidates") or [])))
        ok = [c for c in cands if masks.get(c) is not None]
        if not ok:
            log(f"view {i} ({it.get('label', '')}): no subject found in any candidate; left out")
            continue
        if masks.get(it["path"]) is None:
            it["path"] = ok[0]
            train_imgs[i] = load_rgb(ok[0], max_side)
        if it.get("candidates"):
            it["candidates"] = [c for c in it["candidates"] if masks.get(c) is not None]
        keep.append(i)
    if not any(items[i].get("generated") for i in keep):
        log("no generated view has a subject mask: cameras are found from the whole images")
        return None
    items[:] = [items[i] for i in keep]
    train_imgs[:] = [train_imgs[i] for i in keep]
    return masks


def score_only(req, items, train_imgs, device, torch, env, max_side):
    """Score every generated candidate against the photo (no reconstruction): the app uses this to
    decide whether an angle needs more candidates."""
    a0 = train_imgs[0].shape[1] / train_imgs[0].shape[0]
    same_shape = all(abs(im.shape[1] / im.shape[0] / a0 - 1) < 0.03 for im in train_imgs)
    use_da3 = req.get("pose_engine") == "da3" and req.get("da3_dir") and same_shape
    masks = subject_setup(req, items, max_side, device, torch) if use_da3 else None
    model = try_load_da3(req["da3_dir"], device, torch) if use_da3 else None
    if model is not None:
        def mk(path):
            return masks.get(path) if masks else None

        size = da3_size(train_imgs[0], int(req.get("da3_res", 504)))
        vin = [da3_input(im, size, mk(it["path"])) for im, it in zip(train_imgs, items)]
        sel = select_candidates(items, train_imgs, vin, model, None, torch, max_side,
                                prep=lambda im, path: da3_input(im, size, mk(path)),
                                predict=lambda arrays: da3_predict(model, arrays, torch), min_candidates=1)
        engine = "da3"
    else:
        from vggt.models.vggt import VGGT

        vdir = req["vggt_dir"]
        if os.path.exists(os.path.join(vdir, "config.json")):
            model = VGGT.from_pretrained(vdir)
        else:
            model = VGGT()
            model.load_state_dict(torch.load(os.path.join(vdir, "model.pt"), map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        dtype = (torch.bfloat16 if device == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
        vin = [vggt_input(im) for im in train_imgs]
        sel = select_candidates(items, train_imgs, vin, model, dtype, torch, max_side, min_candidates=1)
        engine = "vggt"
    emit("result", score_only=True, camera_engine=engine, candidate_selection=sel, vram_peak_mib=vram_peak_mib(torch))


def main(req):
    import numpy as np
    import torch

    import splat_trainer as st

    env = torch_env(torch)
    poses_only = bool(req.get("stop_after_poses"))   # CI self-test on machines without a GPU
    device = "cuda" if env["cuda_available"] else "cpu"
    fusion_mode = req.get("assembly") == "fusion"
    if device == "cpu" and not (poses_only or req.get("score_only") or req.get("dry_run")
                                or (fusion_mode and req.get("allow_cpu"))):
        emit("error", code="cuda_unavailable", message="Multi-view reconstruction needs an NVIDIA GPU (CUDA).")
        sys.exit(1)
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    items = req["images"]
    n_views = len(items)
    if len(items) < 2:
        emit("error", code="bad_input", message="Multi-view reconstruction needs at least 2 images.")
        sys.exit(1)
    max_side = int(req.get("max_side", 960))
    progress(0.01, f"loading {len(items)} images")
    train_imgs = [load_rgb(it["path"], max_side) for it in items]
    if req.get("score_only"):
        score_only(req, items, train_imgs, device, torch, env, max_side)
        return
    vin = [vggt_input(im) for im in train_imgs]

    # ------------------------------------------------------------ poses + depth
    engine = req.get("pose_engine") or "vggt"
    a0 = train_imgs[0].shape[1] / train_imgs[0].shape[0]
    same_shape = all(abs(im.shape[1] / im.shape[0] / a0 - 1) < 0.03 for im in train_imgs)
    model = masks = None
    if (engine == "da3" and req.get("da3_dir") and len(items) <= int(req.get("da3_max_views", 32))
            and same_shape):
        masks = subject_setup(req, items, max_side, device, torch)
        if masks:
            masks = keep_masked_views(items, train_imgs, masks, max_side)
        progress(0.03, "loading Depth Anything 3")
        model = try_load_da3(req["da3_dir"], device, torch)
        if model is None:
            masks = None

    def mk(path):
        return masks.get(path) if masks else None

    if model is not None:
        size = da3_size(train_imgs[0], int(req.get("da3_res", 504)))
        vin = [da3_input(im, size, mk(it["path"])) for im, it in zip(train_imgs, items)]
        selection = []
        if any(len(it.get("candidates") or []) > 1 for it in items):
            progress(0.04, "choosing the most consistent generated views")
            selection = select_candidates(items, train_imgs, vin, model, None, torch, max_side,
                                          prep=lambda im, path: da3_input(im, size, mk(path)),
                                          predict=lambda arrays: da3_predict(model, arrays, torch))
        dropped = drop_inconsistent(items, train_imgs, vin, float(req.get("drop_threshold", 0.8)))
        for sel in selection:
            sel["dropped"] = sel["view"] in dropped
        ff_gauss = None
        if req.get("assembly") == "ff":
            c2w_v, K_v, depth, conf, ff_gauss = da3_cameras(model, vin, torch, infer_gs=True)
            log("feed-forward 3D Gaussians from Depth Anything 3" if ff_gauss is not None else
                "this Depth Anything 3 model has no Gaussian head; training from scratch instead")
        else:
            c2w_v, K_v, depth, conf = da3_cameras(model, vin, torch)
        del model
        free(torch)
        metric_engine = bool(req.get("da3_metric", True))
    else:
        if engine == "da3" and not (req.get("da3_dir") and len(items) <= int(req.get("da3_max_views", 32))
                                    and same_shape):
            log("Depth Anything 3 not used (not installed, too many views or mixed image shapes); using VGGT")
        engine = "vggt"
        ff_gauss = None
        c2w_v, K_v, depth, conf, vin, selection = vggt_stage(req, items, train_imgs, vin, device, torch,
                                                              max_side, env)
        metric_engine = False
    log(f"camera engine: {engine}")
    n_views = len(items)            # after dropping inconsistent generated views
    scale = None
    if req.get("moge_path") and not (engine == "da3" and metric_engine):
        progress(0.32, "metric scale (MoGe-2)")
        try:
            scale = metric_scale(req["moge_path"], vin, depth, conf, torch, device=device)
        except Exception as e:  # noqa: BLE001 - metric scale is optional
            log(f"metric scale failed: {e}")
    if scale and np.isfinite(scale) and scale > 0:
        log(f"metric scale factor {scale:.4f}")
        depth = [d * scale for d in depth]
        for m in c2w_v:
            m[:3, 3] *= scale
    metric = bool(scale) or (engine == "da3" and metric_engine)

    if fusion_mode and not poses_only:
        if not req.get("moge_path"):
            emit("error", code="model_missing", message="Sharp fusion needs the MoGe-2 model.")
            sys.exit(1)
        outputs, count, prov, mesh_info = fuse_views(req, items, vin, c2w_v, K_v, depth, conf, device, torch,
                                                     out_dir)
        emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), metric=metric,
             metric_scale_factor=scale, views=n_views, camera_engine=engine, real_views=sum(1 for it in items if not it.get("generated")),
             reference_psnr_db=None, splats=count, mesh=mesh_info, assembly="fusion", candidate_selection=selection,
             provenance={"observed": 0, "inferred": int((prov == 1).sum()), "generative": int((prov == 2).sum())})
        return

    # ------------------------------------------------------------ depth priors (sparse views)
    n = len(items)
    Ks = np.stack([to_original_K(K_v[i], vin[i][1], 0) for i in range(n)])
    subj = [mk(it["path"]) for it in items]           # subject masks of the views kept (None: whole image)
    priors = None
    # Per-view MoGe-2 depth is sharper, but each generated view gets its own depth that does not agree
    # with the others' and the 3D is pulled apart (owner's tests, rc19-rc30): off unless asked for. The
    # engine's joint multi-view depth (DA3 / VGGT) is used instead - smoother, but consistent.
    if (req.get("depth_prior", False) and req.get("moge_path") and n <= int(req.get("prior_max_views", 24))
            and not poses_only):
        try:
            priors = mono_priors(req["moge_path"], train_imgs, vin, depth, conf, device, torch,
                                 align_masks=subj if masks else None)
            if masks:   # a generated view's background is not a camera move of the photo's: unknown
                priors = [p if p is None or not it.get("generated") or subj[i] is None
                          else np.where(resize_map(subj[i].astype(np.uint8), p.shape[1], p.shape[0]) > 0, p, 0.0)
                          .astype(np.float32) for i, (p, it) in enumerate(zip(priors, items))]
            log("MoGe-2 depth priors: " + ", ".join("-" if p is None else "ok" for p in priors))
        except Exception as e:  # noqa: BLE001 - priors are an improvement, not a requirement
            log(f"depth priors unavailable: {e}")
            priors = None

    # subject mode: the engine saw a grey background in the photo, so its depth there means nothing; the
    # photo's background depth comes from MoGe-2 (one view: nothing to disagree with), scaled on the subject
    if masks and subj[0] is not None and req.get("moge_path") and not poses_only:
        try:
            p0 = priors[0] if priors is not None and priors[0] is not None else \
                mono_priors(req["moge_path"], train_imgs[:1], vin[:1], depth[:1], conf[:1], device, torch,
                            align_masks=subj[:1])[0]
            if p0 is not None:
                h0, w0 = train_imgs[0].shape[:2]
                m0 = resize_map(subj[0].astype(np.uint8), w0, h0).astype(bool)
                eng0 = vggt_on_image(depth[0], vin[0][1], h0, w0)[0]
                photo_depth = np.where(m0, eng0, p0).astype(np.float32)
                priors = priors if priors is not None else [None] * n
                priors[0] = photo_depth if not req.get("depth_prior", False) else priors[0]
                log("photo background depth from MoGe-2 (scaled on the subject)")
        except Exception as e:  # noqa: BLE001
            log(f"photo background depth unavailable: {e}")

    # ------------------------------------------------------------ initial points
    progress(0.35, "building the initial point cloud")
    per_view = max(2000, int(req.get("init_points", 400000) * 1.5 / n))
    rng = np.random.default_rng(0)
    pts, cols = [], []
    K_crop = []
    for i in range(n):
        a = vin[i][0]
        Kc = K_v[i]
        K_crop.append(Kc)
        if priors is not None and priors[i] is not None:
            # dense and sharp: the aligned MoGe-2 depth at training resolution
            d = priors[i]
            yy, xx = np.nonzero(d > 0)
            if len(yy) > per_view:
                sel = rng.choice(len(yy), per_view, replace=False)
                yy, xx = yy[sel], xx[sel]
            pts.append(unproject(d, Ks[i], c2w_v[i])[yy, xx])
            cols.append(train_imgs[i][yy, xx].astype(np.float32) / 255.0)
            continue
        d, c = depth[i], conf[i]
        # confidence scales differ between engines (VGGT >= 1, DA3 not): keep the better 65 %
        ok = (d > 0) & np.isfinite(d) & (c >= np.percentile(c, 35))
        if items[i].get("generated") and vin[i][1].get("mask") is not None:
            ok &= vin[i][1]["mask"]
        yy, xx = np.nonzero(ok)
        if len(yy) > per_view:
            sel = rng.choice(len(yy), per_view, replace=False)
            yy, xx = yy[sel], xx[sel]
        pts.append(unproject(d, Kc, c2w_v[i])[yy, xx])
        cols.append(a[yy, xx])
    points = np.concatenate(pts).astype(np.float32)
    colors = np.concatenate(cols).astype(np.float32)
    log(f"{len(points):,} initial points from {n} views")
    if poses_only:
        emit("result", outputs={}, poses_only=True, views=n, points=int(len(points)), metric_scale_factor=scale,
             candidate_selection=selection,
             c2ws=[m.tolist() for m in c2w_v], vram_peak_mib=vram_peak_mib(torch))
        return

    # ------------------------------------------------------------ training
    c2ws = np.stack(c2w_v)
    weights = [float(it.get("weight", 1.0)) for it in items]
    pixel_w = None
    if any(it.get("generated") for it in items):
        # depth for the photo-consistency check: MoGe-2 priors when enabled, else the engine's joint depth
        def view_depth(i):
            if priors is not None and priors[i] is not None:
                return priors[i]
            h, w = train_imgs[i].shape[:2]
            return vggt_on_image(depth[i], vin[i][1], h, w)[0]

        pixel_w = [None] * n
        d0 = view_depth(0)
        for i, it in enumerate(items):
            if it.get("generated"):
                pixel_w[i] = consistency_map(train_imgs[0], d0, Ks[0], c2ws[0], train_imgs[i], view_depth(i),
                                             Ks[i], c2ws[i])
                log(f"view {i} ({it.get('label', '')}): {float((pixel_w[i] < 0.5).mean()):.0%} of its pixels "
                    "contradict the photo and are down-weighted")
    if masks and any(it.get("generated") and subj[i] is not None for i, it in enumerate(items)):
        pixel_w = pixel_w or [None] * n
        for i, it in enumerate(items):
            if it.get("generated") and subj[i] is not None:
                h, w = train_imgs[i].shape[:2]
                m = resize_map(subj[i].astype(np.uint8), w, h).astype(np.float32)
                pixel_w[i] = m if pixel_w[i] is None else pixel_w[i] * m
    ff_init = None
    if ff_gauss is not None:
        if scale and np.isfinite(scale) and scale > 0:      # MoGe metric scale applied to the cameras
            ff_gauss = dict(ff_gauss, means=ff_gauss["means"] * scale, scales=ff_gauss["scales"] * scale)
        ff_init = ff_init_gaussians(ff_gauss, items, vin, depth, subj,
                                    priors[0] if priors is not None else None, train_imgs[0], Ks[0], c2ws[0])
        ff_gauss = None
    if req.get("dry_run"):   # tests: everything up to the GPU training, on any device
        emit("result", dry_run=True, camera_engine=engine, views=n, points=int(len(points)), metric=metric,
             ff_gaussians=None if ff_init is None else int(len(ff_init["means"])),
             priors=[p is not None for p in (priors or [None] * n)],
             down_weighted=[None if m is None else float((m < 0.5).mean()) for m in (pixel_w or [None] * n)],
             subject_mode=bool(masks), candidate_selection=selection, c2ws=[m.tolist() for m in c2ws])
        return
    cfg = st.TrainConfig(steps=int(req.get("steps", 10000)), sh_degree=int(req.get("sh_degree", 3)),
                         max_gaussians=int(req.get("max_gaussians", 2_000_000)),
                         init_points=int(req.get("init_points", 400000)))
    ff_raw = None
    if ff_init is not None:
        # the feed-forward splat is already coherent: a short, gentle polish (positions move 10x
        # slower) recovers the photo's detail without letting the generated views tear it apart again
        cfg.steps = int(req.get("ff_refine_steps", 3000))
        cfg.lr_means *= 0.1
        cfg.coarse_until = 0.0
        cfg.max_gaussians = max(cfg.max_gaussians, int(len(ff_init["means"]) * 1.5))
        try:     # the raw feed-forward result is kept for comparison
            raw = st.params_from_gaussians(ff_init, cfg, "cuda")
            ff_raw = os.path.join(out_dir, "scene_feedforward.ply")
            st.write_ply(raw, ff_raw)
            sizes0 = [(im.shape[1], im.shape[0]) for im in train_imgs]
            zc, yaw = turntable_orbit(raw["means"].detach().float().cpu().numpy(), c2ws, Ks[0], sizes0[0],
                                      subj[0])
            render_turntable(raw, Ks[0], sizes0[0], zc, os.path.join(out_dir, "turntable_feedforward"),
                             cfg.sh_degree, torch, base=c2ws[0], yaw_deg=yaw)
            del raw
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - a comparison only
            log(f"raw feed-forward export failed: {e}")
            ff_raw = None
    # Cap the allocator below the card's size: on Windows an over-full GPU silently spills into
    # shared system memory (10-50x slower); an out-of-memory error lets us retry smaller instead.
    torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.94)))
    # generated views may drift in exposure / white balance: learn a colour correction for each
    appearance = [bool(it.get("generated")) for it in items] if req.get("appearance", True) else None
    # ... and their estimated cameras are slightly off: refine them while training (the photo stays fixed)
    # (real photos too, except the first one, which anchors the scene; not for long videos, where the
    # engine's poses from many overlapping frames are already well constrained)
    pose_opt = None
    if req.get("pose_refine", True):
        pose_opt = [k > 0 and (bool(it.get("generated")) or n <= 32) for k, it in enumerate(items)]
    for attempt in range(3):
        oom = False
        app_out, pose_out = [], []
        try:
            params, hist = st.train(train_imgs, c2ws, Ks, weights, points, colors, cfg, device="cuda",
                                    progress=lambda v, m: progress(0.37 + 0.5 * v, m),
                                    pixel_weights=pixel_w, depth_priors=priors,
                                    appearance=appearance if appearance and any(appearance) else None,
                                    appearance_out=app_out,
                                    pose_opt=pose_opt if pose_opt and any(pose_opt) else None,
                                    pose_out=pose_out, init=ff_init)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or attempt == 2:
                raise
            oom = True
        if oom:   # outside the except block: the traceback no longer pins the failed attempt's tensors
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            from PIL import Image

            train_imgs = [np.asarray(Image.fromarray(im).resize((max(8, im.shape[1] * 3 // 4),
                                                                 max(8, im.shape[0] * 3 // 4)), Image.LANCZOS))
                          for im in train_imgs]
            sizes_now = [(im.shape[1], im.shape[0]) for im in train_imgs]
            if priors is not None:
                priors = [resize_map(p, *sz) for p, sz in zip(priors, sizes_now)]
            if pixel_w is not None:
                pixel_w = [resize_map(p, *sz) for p, sz in zip(pixel_w, sizes_now)]
            Ks = Ks.copy()
            Ks[:, :2] *= 0.75
            cfg.max_gaussians = int(cfg.max_gaussians * 0.6)
            log(f"out of GPU memory while training; retrying with smaller images "
                f"({train_imgs[0].shape[1]}x{train_imgs[0].shape[0]}) and at most {cfg.max_gaussians:,} splats")

    pose_refined = []
    if pose_out:
        for i, m in enumerate(pose_out):
            if pose_opt and pose_opt[i] and items[i].get("generated"):
                c0, c1 = c2ws[i][:3, :3], m[:3, :3]
                ang = float(np.degrees(np.arccos(np.clip((np.trace(c0.T @ c1) - 1) / 2, -1, 1))))
                pose_refined.append({"view": items[i].get("label") or os.path.basename(items[i]["path"]),
                                     "rotation_deg": round(ang, 2)})
        c2ws = np.stack(pose_out)
        if pose_refined:
            log("refined cameras of generated views: " +
                ", ".join(f"{p['view']} {p['rotation_deg']}°" for p in pose_refined))

    # ------------------------------------------------------------ cleanup
    sizes = [(im.shape[1], im.shape[0]) for im in train_imgs]
    if req.get("cleanup", True):
        progress(0.87, "removing floaters and splats no view has seen")
        with torch.no_grad():
            keep = st.cleanup_mask(params, c2ws, Ks, sizes)
        removed = int((~keep).sum())
        if 0 < removed < len(keep):
            params = st.subset(params, keep)
            log(f"cleanup: removed {removed:,} of {len(keep):,} splats (unseen, transparent, oversized or isolated)")

    # ------------------------------------------------------------ turntable preview
    turntable = None
    if req.get("turntable", True):
        try:
            progress(0.875, "rendering a turntable preview")
            zc, yaw = turntable_orbit(params["means"].detach().float().cpu().numpy(), c2ws, Ks[0], sizes[0],
                                      subj[0])
            log(f"turntable: ±{yaw:.0f}° around a point {zc:.2f} in front of the photo's camera")
            turntable = os.path.join(out_dir, "turntable")
            render_turntable(params, Ks[0], sizes[0], zc, turntable, cfg.sh_degree, torch, base=c2ws[0],
                             yaw_deg=yaw)
        except Exception as e:  # noqa: BLE001 - a preview only
            log(f"turntable preview failed: {e}")
            turntable = None

    # ------------------------------------------------------------ provenance
    progress(0.88, "labelling observed / inferred / generated splats")
    real = [i for i, it in enumerate(items) if not it.get("generated")]
    if len(real) > 60:
        real = [real[int(k)] for k in np.linspace(0, len(real) - 1, 60).round()]
    vm = torch.linalg.inv(torch.as_tensor(c2ws, dtype=torch.float32)).cuda()
    Kt = torch.as_tensor(Ks, dtype=torch.float32).cuda()

    def render_depth(v):
        with torch.no_grad():
            out, alpha, _ = st.gsplat_render(params, vm[v:v + 1], Kt[v:v + 1], sizes[v][0], sizes[v][1],
                                             0, mode="ED")
        d = out[..., 0]
        return torch.where(alpha[..., 0] > 0.5, d, torch.full_like(d, float("inf")))

    counts = st.visibility_counts(params, c2ws, Ks, sizes, real, render_depth)
    any_gen = any(it.get("generated") for it in items)
    codes = st.provenance_codes(counts, any_gen)

    # quality on the reference view
    with torch.no_grad():
        rgb, _, _ = st.gsplat_render(params, vm[:1], Kt[:1], sizes[0][0], sizes[0][1], cfg.sh_degree)
        ref_psnr = st.psnr(rgb[..., :3].clamp(0, 1), torch.from_numpy(train_imgs[0]).cuda().float() / 255)
    ply = os.path.join(out_dir, "scene.ply")
    count = st.write_ply(params, ply, codes)
    log(f"{count:,} splats; reference-view PSNR {ref_psnr:.2f} dB")

    cams = []
    for i, it in enumerate(items):
        cams.append({"width": sizes[i][0], "height": sizes[i][1], "fx": float(Ks[i][0, 0]),
                     "fy": float(Ks[i][1, 1]), "cx": float(Ks[i][0, 2]), "cy": float(Ks[i][1, 2]),
                     "c2w": c2ws[i].tolist(), "file": os.path.basename(it["path"]),
                     "generated": bool(it.get("generated"))})
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cams, f)
    outputs = {"ply": ply, "cameras": os.path.join(out_dir, "cameras.json")}
    if turntable:
        outputs["turntable_frames"] = turntable
    if ff_raw:
        outputs["ply_feedforward"] = ff_raw
        tf = os.path.join(out_dir, "turntable_feedforward")
        if os.path.isdir(tf):
            outputs["turntable_feedforward_frames"] = tf

    # the mesh is fused from the TRAINED splat's depth and colour in every view (same volume as the
    # splat, generated angles included); the raw engine depth of the real views is the fallback
    mesh_views = None
    if req.get("mesh", True):
        try:
            progress(0.92, "rendering depth for the mesh")
            mesh_views = splat_depth_views(params, c2ws, Ks, sizes, torch, max_views=60)
        except Exception as e:  # noqa: BLE001
            log(f"rendered depth for the mesh failed, using the engine depth: {e}")
    params = None  # free GPU memory before meshing
    torch.cuda.empty_cache()
    mesh_info = None
    if req.get("mesh", True):
        progress(0.93, "mesh (TSDF fusion)")
        try:
            if mesh_views:
                dd, cc, mK, mC = mesh_views
            else:
                use = [i for i in range(n) if not items[i].get("generated")] or list(range(n))
                use = use[:: max(1, len(use) // 60)]
                dd = [depth[i] for i in use]
                cc = [(vin[i][0] * 255).astype(np.uint8) for i in use]
                mK, mC = [K_crop[i] for i in use], [c2ws[i] for i in use]
            med = float(np.median([np.median(d[d > 0]) for d in dd if np.any(d > 0)]))
            mesh_info = tsdf_mesh(dd, mK, mC, cc, os.path.join(out_dir, "scene.obj"), max(med / 250, 1e-3))
            mesh_info["source"] = "trained splat" if mesh_views else "engine depth"
            outputs["obj"] = os.path.join(out_dir, "scene.obj")
        except Exception as e:  # noqa: BLE001 - the mesh is optional
            log(f"mesh export failed: {e}")
    counts_np = codes.cpu().numpy()
    emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), metric=metric, camera_engine=engine,
         metric_scale_factor=scale, views=n, real_views=sum(1 for it in items if not it.get("generated")),
         reference_psnr_db=round(ref_psnr, 2), splats=count, mesh=mesh_info, loss_history=hist[-20:],
         candidate_selection=selection,
         pose_refinement=pose_refined, subject_mode=bool(masks),
         assembly="ff" if ff_init is not None else "train",
         colour_correction=[None if a is None else {"gain": [round(a[0][c][c], 3) for c in range(3)],
                                                    "offset": [round(x, 3) for x in a[1]]} for a in app_out],
         provenance={"observed": int((counts_np == 0).sum()), "inferred": int((counts_np == 1).sum()),
                     "generative": int((counts_np == 2).sum())})


if __name__ == "__main__":
    run(main)
