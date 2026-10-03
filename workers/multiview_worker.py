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
        h, w = d.shape
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(col)), o3d.geometry.Image(np.ascontiguousarray(d, np.float32)),
            depth_scale=1.0, depth_trunc=float(np.percentile(d, 98)), convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(c2w))
    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    # OBJ viewers expect y up: flip OpenCV (y down, z forward)
    mesh.transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    o3d.io.write_triangle_mesh(out_obj, mesh, write_vertex_colors=True)
    return {"vertices": len(mesh.vertices), "faces": len(mesh.triangles), "voxel_m": voxel}


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
        r = w / (info["sx"] * VGGT_W)
        xs = ((np.arange(w) + 0.5) / r / info["sx"]).astype(int)
        ys = ((np.arange(h) + 0.5) / r / info["sy"] - info["crop"]).astype(int)
        inside_y = (ys >= 0) & (ys < depth[i].shape[0])
        xs = np.clip(xs, 0, depth[i].shape[1] - 1)
        yc = np.clip(ys, 0, depth[i].shape[0] - 1)
        zv = depth[i][yc][:, xs]
        cv = conf[i][yc][:, xs]
        ok = valid & inside_y[:, None] & (cv > np.percentile(conf[i], 50))
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


def rigidity_score(model, ref_vin, cand_vin, dtype, torch, target_deg=None):
    """How well a generated view is explained as a pure camera move of the reference photo.

    VGGT poses the pair and gives the photo's depth; the photo is reprojected into the candidate's
    view and compared where both see the same surface. A changed pose, head turn or expression
    shows up as colour mismatch. Lower is better. Returns (score, colour error, angle°, overlap)."""
    import math

    import numpy as np

    p = run_vggt(model, [ref_vin[0], cand_vin[0]], dtype, torch)
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
    rel = w2c1 @ c2w0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel[:3, :3]) - 1) / 2))))
    overlap = float(ok.mean())
    if ok.sum() < 500:
        return 10.0, None, angle, overlap
    a = ref_vin[0].reshape(-1, 3)[ok]
    b = cand_vin[0][vi[ok], ui[ok]]
    diff = np.abs(a - b).mean(1)
    # Changes are local (a turned head, a moved hand): the share of clearly mismatching pixels catches
    # them, which a median would ignore; the mean error adds the overall fit.
    err = float((diff > 0.12).mean() + diff.mean())
    score = err + 0.5 * max(0.0, 0.25 - overlap)          # little shared surface: less trustworthy
    if target_deg:
        score += 0.003 * max(0.0, abs(angle - target_deg) - 15)   # it must really have moved
    return score, err, angle, overlap


def select_candidates(items, train_imgs, vin, model, dtype, torch, max_side):
    """For every view with several generated candidates keep the most rigid one (rigidity_score)."""
    report = []
    for i, it in enumerate(items):
        cands = it.get("candidates") or []
        if len(cands) < 2:
            continue
        scored = []
        for path in cands:
            img = load_rgb(path, max_side)
            cv = vggt_input(img)
            sc, err, ang, ov = rigidity_score(model, vin[0], cv, dtype, torch, it.get("target_deg"))
            scored.append((sc, path, img, cv, err, ang, ov))
            log(f"view {i} ({it.get('label', '')}) candidate {os.path.basename(path)}: score {sc:.4f} "
                f"(colour error {err if err is None else round(err, 4)}, camera moved {ang:.1f}°, "
                f"shared surface {ov:.0%})")
        best = min(scored, key=lambda t: t[0])
        items[i] = dict(it, path=best[1])
        train_imgs[i], vin[i] = best[2], best[3]
        report.append({"view": it.get("label", str(i)), "chosen": os.path.basename(best[1]),
                       "scores": {os.path.basename(t[1]): round(t[0], 4) for t in scored},
                       "camera_moved_deg": round(best[5], 1)})
        log(f"view {i} ({it.get('label', '')}): kept {os.path.basename(best[1])}")
    return report


def main(req):
    import numpy as np
    import torch

    import splat_trainer as st

    env = torch_env(torch)
    poses_only = bool(req.get("stop_after_poses"))   # CI self-test on machines without a GPU
    device = "cuda" if env["cuda_available"] else "cpu"
    fusion_mode = req.get("assembly") == "fusion"
    if device == "cpu" and not (poses_only or (fusion_mode and req.get("allow_cpu"))):
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
    vin = [vggt_input(im) for im in train_imgs]

    # ------------------------------------------------------------ poses + depth
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
    c2w_v, K_v, depth, conf = estimate_cameras(vin, model, dtype, torch, chunk, int(req.get("overlap", 8)))
    del model
    free(torch)

    scale = None
    if req.get("moge_path"):
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

    if fusion_mode and not poses_only:
        if not req.get("moge_path"):
            emit("error", code="model_missing", message="Sharp fusion needs the MoGe-2 model.")
            sys.exit(1)
        outputs, count, prov, mesh_info = fuse_views(req, items, vin, c2w_v, K_v, depth, conf, device, torch,
                                                     out_dir)
        emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), metric=bool(scale),
             metric_scale_factor=scale, views=n_views, real_views=sum(1 for it in items if not it.get("generated")),
             reference_psnr_db=None, splats=count, mesh=mesh_info, assembly="fusion", candidate_selection=selection,
             provenance={"observed": 0, "inferred": int((prov == 1).sum()), "generative": int((prov == 2).sum())})
        return

    # ------------------------------------------------------------ initial points
    progress(0.35, "building the initial point cloud")
    n = len(items)
    per_view = max(2000, int(req.get("init_points", 400000) * 1.5 / n))
    rng = np.random.default_rng(0)
    pts, cols = [], []
    K_crop = []
    for i in range(n):
        a = vin[i][0]
        Kc = K_v[i]
        K_crop.append(Kc)
        d, c = depth[i], conf[i]
        ok = (d > 0) & np.isfinite(d) & (c > max(1.0, np.percentile(c, 35)))
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
    Ks = np.stack([to_original_K(K_v[i], vin[i][1], 0) for i in range(n)])
    c2ws = np.stack(c2w_v)
    weights = [float(it.get("weight", 1.0)) for it in items]
    cfg = st.TrainConfig(steps=int(req.get("steps", 10000)), sh_degree=int(req.get("sh_degree", 3)),
                         max_gaussians=int(req.get("max_gaussians", 2_000_000)),
                         init_points=int(req.get("init_points", 400000)))
    # Cap the allocator below the card's size: on Windows an over-full GPU silently spills into
    # shared system memory (10-50x slower); an out-of-memory error lets us retry smaller instead.
    torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.94)))
    for attempt in range(3):
        try:
            params, hist = st.train(train_imgs, c2ws, Ks, weights, points, colors, cfg, device="cuda",
                                    progress=lambda v, m: progress(0.37 + 0.5 * v, m))
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or attempt == 2:
                raise
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            from PIL import Image

            train_imgs = [np.asarray(Image.fromarray(im).resize((max(8, im.shape[1] * 3 // 4),
                                                                 max(8, im.shape[0] * 3 // 4)), Image.LANCZOS))
                          for im in train_imgs]
            Ks = Ks.copy()
            Ks[:, :2] *= 0.75
            cfg.max_gaussians = int(cfg.max_gaussians * 0.6)
            log(f"out of GPU memory while training; retrying with smaller images "
                f"({train_imgs[0].shape[1]}x{train_imgs[0].shape[0]}) and at most {cfg.max_gaussians:,} splats")

    # ------------------------------------------------------------ provenance
    progress(0.88, "labelling observed / inferred / generated splats")
    real = [i for i, it in enumerate(items) if not it.get("generated")]
    if len(real) > 60:
        real = [real[int(k)] for k in np.linspace(0, len(real) - 1, 60).round()]
    sizes = [(im.shape[1], im.shape[0]) for im in train_imgs]
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

    params = None  # free GPU memory before meshing
    torch.cuda.empty_cache()
    mesh_info = None
    if req.get("mesh", True):
        progress(0.93, "mesh (TSDF fusion)")
        try:
            use = [i for i in range(n) if not items[i].get("generated")] or list(range(n))
            use = use[:: max(1, len(use) // 60)]
            med = float(np.median([np.median(depth[i][depth[i] > 0]) for i in use]))
            dd = [depth[i] for i in use]
            cc = [(vin[i][0] * 255).astype(np.uint8) for i in use]
            mesh_info = tsdf_mesh(dd, [K_crop[i] for i in use], [c2ws[i] for i in use], cc,
                                  os.path.join(out_dir, "scene.obj"), max(med / 250, 1e-3))
            outputs["obj"] = os.path.join(out_dir, "scene.obj")
        except Exception as e:  # noqa: BLE001 - the mesh is optional
            log(f"mesh export failed: {e}")
    counts_np = codes.cpu().numpy()
    emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), metric=bool(scale),
         metric_scale_factor=scale, views=n, real_views=sum(1 for it in items if not it.get("generated")),
         reference_psnr_db=round(ref_psnr, 2), splats=count, mesh=mesh_info, loss_history=hist[-20:],
         candidate_selection=selection,
         provenance={"observed": int((counts_np == 0).sum()), "inferred": int((counts_np == 1).sum()),
                     "generative": int((counts_np == 2).sum())})


if __name__ == "__main__":
    run(main)
