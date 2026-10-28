"""Apple SHARP with an injected depth (runtime 'photo-cu128') - Real 3D, "the photo locked" (rc56).

SHARP's own depth is monocular with a guessed lens (30 mm without EXIF), so its scale and volume are
arbitrary. Its network can align its depth to a given one: ``RGBGaussianPredictor.forward(image,
disparity_factor, depth=...)`` runs a learned UNet (``LearnedAlignment``, part of the released checkpoint)
that turns the given depth into a smooth local scale map for SHARP's two depth layers. The CLI never passes
it; this worker does, with MoGe's metric depth of the photo at the photo's real lens.

Request (mode "photo"):
  {"mode": "photo", "checkpoint": ".../sharp_2572gikvuh.pt", "geometry": ".../photo.npz" (moge_worker output:
   points, mask, intrinsics, image), "output_dir": "...", "face_model": ".../yunet.onnx" | null,
   "ipd_target_m": 0.063, "human_seg": ".../sapiens2_0.4b_seg.safetensors" | null,
   "human_pointmap": ".../sapiens2_1b_pointmap.safetensors" | null, "plain": true, "allow_cpu": false}
Writes output_dir/locked.ply (SHARP PLY with metadata) and, with "plain", output_dir/plain.ply (SHARP alone,
for comparison). Result: scale diagnostics (how far SHARP's own depth was from MoGe's, the scale used).

rc58: people's shape from Sapiens2 (human_geometry.py) inside their outline, and the real-world scale always
calibrated - to the adult eye distance (ipd_target_m) when faces are found, else to the human model's
distance of the people.

Request (mode "fill", rc57): new content for the holes a moved camera sees, tied to the known geometry:
  {"mode": "fill", "checkpoint": "...", "moge_path": ".../model.pt", "moge_version": "v3" | "v2",
   "views": [{"image": "filled.png", "mask": "hole.png" (white = hole), "known": "known.npz" (depth: the
              rendered depth of the scene, inf where it has nothing), "fx", "fy", "cx", "cy",
              "out": "fill.ply"}], "ring_px": 12}
For every view: MoGe's depth of the filled image is aligned (robust scale and shift) to the rendered depth
in a ring around the holes - on the far side, the background the hole belongs to - and SHARP runs with the
rendered depth where the scene exists and the aligned depth inside the holes. The app keeps only the
splats that fall in the holes.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

INTERNAL = 1536               # SHARP's working resolution (square)
IPD_TARGET_M = 0.063          # adult interpupillary distance, population median (women ~62, men ~64 mm)


def eye_distances(img, points, model_path, min_face_px=40):
    """3D distance between the eyes of every face found (YuNet, OpenCV), in the units of ``points``.

    rc8: each eye's point is the median of the nearer half of a 5 x 5 window (the face, not the background
    behind its edge), and a face whose two eyes differ in depth more than they are apart sideways (eyes
    sampled across a depth edge, or a face seen from the side) is marked not usable."""
    import cv2

    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.8, 0.3, 50)
    _, faces = det.detect(np.ascontiguousarray(img[..., ::-1]))
    out = []
    for f in [] if faces is None else faces:
        if min(f[2], f[3]) < min_face_px:
            continue
        eyes = []
        for ex, ey in ((f[4], f[5]), (f[6], f[7])):
            x, y = int(round(ex)), int(round(ey))
            win = points[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
            win = win[np.isfinite(win).all(1) & (win[:, 2] > 0)]
            if not len(win):
                break
            win = win[win[:, 2] <= np.median(win[:, 2])]
            eyes.append(np.median(win, 0))
        if len(eyes) == 2:
            d = eyes[0] - eyes[1]
            lateral, dz = float(np.hypot(d[0], d[1])), float(abs(d[2]))
            out.append({"box": [float(v) for v in f[:4]], "score": float(f[14]),
                        "ipd": float(np.linalg.norm(d)), "depth_gap": round(dz, 4),
                        "usable": bool(dz <= 1.2 * lateral)})
    return out


def face_scale(faces, target=IPD_TARGET_M, prior=None):
    """Factor that brings the eye distance of the faces to ``target`` metres (rc58: always, not only
    outside a range - owner: people always came out too big). 1.0 without usable faces; never more than 2x.

    rc8 (owner's photo: one face measured 362 mm, which halved the whole scene): faces whose eyes were not
    measured on the face (eye_distances "usable") or whose distance is impossible even for a depth off by 4x
    (under 2 cm or over 25 cm) are left out; when the rest disagree, the faces that agree with ``prior`` (the
    human model's scale) - else with the biggest face - are used. Marks each face "used" or "rejected"."""
    if not faces:
        return 1.0
    good = []
    for f in faces:
        if not f.get("usable", True):
            f["rejected"] = "eyes not both on the face (depth gap)"
        elif not 0.02 <= f["ipd"] <= 0.25:
            f["rejected"] = f"impossible eye distance ({f['ipd'] * 1000:.0f} mm)"
        else:
            good.append(f)
    if not good:
        return 1.0
    if len(good) > 1 and max(f["ipd"] for f in good) > 1.5 * min(f["ipd"] for f in good):
        ref = target / prior if prior else max(good, key=lambda f: f["box"][2] * f["box"][3])["ipd"]
        agree = [f for f in good if 1 / 1.5 <= f["ipd"] / ref <= 1.5]
        for f in good:
            if f not in agree:
                f["rejected"] = "disagrees with the other cues"
        good = agree or [max(good, key=lambda f: f["box"][2] * f["box"][3])]
    for f in good:
        f["used"] = True
    ipd = float(np.median([f["ipd"] for f in good]))
    return float(min(max(target / ipd, 0.5), 2.0))


def shade(depth, f_px):
    """Grey shaded relief of a depth map (light from the camera's upper left): shapes such as a nose
    show up clearly, to compare two depths of the same picture."""
    import numpy as np

    z = np.where(np.isfinite(depth), depth, np.nan)
    gy, gx = np.gradient(z)
    zz = np.where(np.isfinite(z), z, 1.0)
    n = np.stack([-gx * f_px / zz, -gy * f_px / zz, np.ones_like(zz)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    light = np.array([-0.4, -0.5, 0.77])
    light /= np.linalg.norm(light)
    v = np.nan_to_num(np.clip(n @ light, 0, 1), nan=0.0)
    return (np.where(np.isfinite(z), 40 + 215 * v, 0)).astype(np.uint8)


def people_geometry(req, img, points, valid, f_px, device, torch):
    """Sapiens2's shape of the people put into MoGe's depth. Returns (points, report or None, parts or None);
    parts (for snapping SHARP's splats and the people_shape.png sheet): the people's weight map (1 inside, 0
    outside, a ramp at the outline), their mask and box, and MoGe's own depth."""
    import numpy as np

    import human_geometry as hg

    progress(0.05, "finding the people (Sapiens2)")
    net, arch = hg.load(req["human_seg"], "seg", device, torch)
    people = hg.people_mask(net, img, torch, device)
    del net
    torch.cuda.empty_cache() if device.type == "cuda" else None
    box = hg.crop_box(people)
    if box is None:
        log("no person found: the scene's own depth is used")
        return points, None, None
    progress(0.12, "people's 3D shape (Sapiens2)")
    net, arch = hg.load(req["human_pointmap"], "pointmap", device, torch)
    d_crop, own_f = hg.pointmap_depth(net, img, box, f_px, torch, device)
    del net
    torch.cuda.empty_cache() if device.type == "cuda" else None
    y0, y1, x0, x1 = box
    human = np.full(img.shape[:2], np.nan)
    human[y0:y1, x0:x1] = d_crop
    z = points[..., 2]
    fused, ratio = hg.fuse_people(np.where(valid, z, np.nan), valid, human, people)
    if ratio is None:
        log("the people are too small in the picture for their own 3D shape")
        return points, None, None
    k = np.where(valid & np.isfinite(fused) & (z > 0), fused / np.where(z > 0, z, 1.0), 1.0)
    out = points * k[..., None]
    rep = {"people_fraction": round(float(people.mean()), 4), "box": [int(v) for v in box],
           "scene_vs_human_distance": round(ratio, 4), "human_model": arch,
           "human_model_lens_px": round(own_f, 1), "lens_px": round(f_px, 1)}
    log(f"people's shape from Sapiens2 ({people.mean():.0%} of the picture); the scene's depth put them "
        f"{ratio:.3f}x as far as the human model does")
    both = valid & people & np.isfinite(human) & (human > 0)
    parts = {"weight": hg.inner_weight(both, ramp=6), "mask": people, "box": box, "scene_z": z.copy()}
    return out, rep, parts


def people_sheet(path, img, parts, depths, f_px):
    """people_shape.png: the picture of the people, then each depth map (H, W) shaded so the relief shows."""
    import numpy as np
    from PIL import Image

    y0, y1, x0, x1 = parts["box"]
    m = np.where(parts["mask"][y0:y1, x0:x1], 1.0, np.nan)
    panels = [img[y0:y1, x0:x1]]
    for d in depths:
        panels.append(np.repeat(shade(d[y0:y1, x0:x1] * m, f_px)[..., None], 3, -1))
    Image.fromarray(np.concatenate(panels, 1).astype(np.uint8)).save(path)
    return path


def front_depth(means, h, w, torch):
    """The depth of SHARP's front splats as an (h, w) map of the picture."""
    import torch.nn.functional as F

    g = SHARP_GRID
    zf = means[: g * g, 2].reshape(1, 1, g, g).float()
    return F.interpolate(zf, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def target_depth(z, valid, mono, torch):
    """The depth SHARP is aligned to: the given metric depth where valid, elsewhere SHARP's own depth brought
    to the same scale (its alignment UNet needs a value everywhere)."""
    ratio = torch.median(z[valid] / mono[valid]) if valid.any() else torch.tensor(1.0, device=z.device)
    return torch.where(valid, z, mono * ratio), float(ratio)


def load_predictor(path, device, torch):
    from sharp.models import PredictorParams, create_predictor

    p = create_predictor(PredictorParams())
    p.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return p.eval().to(device)


def predict(predictor, img, f_px, device, torch, depth=None, valid=None, cx=None, cy=None):
    """SHARP on an image (H,W,3 uint8) with focal length f_px; ``depth`` (H,W metric, ``valid`` mask) is
    injected when given. ``cx, cy``: the principal point in the image's pixels when it is a crop of a larger
    picture (default its centre), so the splats land in the larger picture's camera frame. Returns
    (gaussians in metric camera space, diagnostics)."""
    import numpy as np
    import torch.nn.functional as F
    from sharp.utils.gaussians import unproject_gaussians

    h, w = img.shape[:2]
    image = torch.from_numpy(img.copy()).float().to(device).permute(2, 0, 1)[None] / 255.0
    image = F.interpolate(image, size=(INTERNAL, INTERNAL), mode="bilinear", align_corners=True)
    disparity_factor = torch.tensor([f_px / w]).float().to(device)
    info = {}
    tgt = None
    with torch.no_grad():
        if depth is not None:
            # the alignment UNet compares the given depth with SHARP's own, pixel for pixel: same resolution
            mono = disparity_factor[:, None, None, None] / predictor.monodepth_model(image).disparity[:, 0:1].clamp(
                min=1e-4, max=1e4)
            size = tuple(mono.shape[-2:])
            z = torch.from_numpy(np.nan_to_num(depth, nan=0.0, posinf=0.0)).float().to(device)[None, None]
            ok = torch.from_numpy(valid.astype("float32")).to(device)[None, None]
            z = F.interpolate(z, size=size, mode="nearest")
            ok = (F.interpolate(ok, size=size, mode="nearest") > 0.5) & (z > 0)
            tgt, ratio = target_depth(z, ok, mono, torch)
            info = {"sharp_own_scale": round(1.0 / ratio, 4), "depth_coverage": round(float(ok.float().mean()), 4)}
        g = predictor(image, disparity_factor, depth=tgt)
        px, py = (w / 2 if cx is None else cx), (h / 2 if cy is None else cy)
        K = torch.tensor([[f_px, 0, px, 0], [0, f_px, py, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().to(device)
        K[0] *= INTERNAL / w
        K[1] *= INTERNAL / h
        g = unproject_gaussians(g, torch.eye(4).to(device), K, (INTERNAL, INTERNAL))
    return g, info


SHARP_GRID = 768          # SHARP: 2 layers of 768 x 768 pixel-aligned splats over its (stretched) input


def to_grid(a, torch, g=SHARP_GRID):
    """An (H, W) map (NaN = unknown) resampled to SHARP's g x g splat grid (the image stretched to a square,
    as SHARP sees it). Returns (values, known share) as torch tensors on a's device."""
    import torch.nn.functional as F

    a = torch.as_tensor(a, dtype=torch.float32)
    ok = torch.isfinite(a).float()
    num = F.interpolate((torch.nan_to_num(a, nan=0.0) * ok)[None, None], size=(g, g), mode="bilinear",
                        antialias=True, align_corners=False)[0, 0]
    den = F.interpolate(ok[None, None], size=(g, g), mode="bilinear", antialias=True, align_corners=False)[0, 0]
    return num / den.clamp(min=1e-6), den


def to_grid_nearest(a, torch, g=SHARP_GRID):
    """An (H, W) map sampled at SHARP's grid cell centres (no averaging: a depth edge stays an edge)."""
    import torch.nn.functional as F

    a = torch.as_tensor(a, dtype=torch.float32)
    return F.interpolate(a[None, None], size=(g, g), mode="nearest")[0, 0]


def snap_to_depth(g, depth, weight, torch, lo=0.6, hi=1.6, band=None, back=None):
    """Put SHARP's splats exactly at ``depth`` (H, W metres, NaN = unknown) where ``weight`` (H, W in [0, 1])
    says so, blending to SHARP's own placement where it falls to 0.

    SHARP only follows an injected depth through its learned alignment, which is smooth: the fine relief of
    a person (nose, face, folds of clothes) from the human model is mostly lost, more so on soft video frames
    whose own SHARP depth is flat. Each pixel's front splat is moved along its line of sight to the given
    depth, and the splat behind it by the same factor (sizes too, so the picture from the camera does not
    change). ``band`` (H, W bool, rc7): pixels next to a depth edge, already sent to one side
    (precision.resolve_edges): there the depth is taken at the cell centre instead of averaged across the
    edge, and the splat behind goes to ``back`` (the far side's depth). Returns (means, scales, front factor)
    for all 2 x 768 x 768 splats."""
    gsz = SHARP_GRID
    means, scales = g.mean_vectors[0], g.singular_values[0]
    dev = means.device
    t, known = to_grid(depth, torch)
    w, _ = to_grid(weight, torch)
    t, known, w = t.to(dev), known.to(dev), w.to(dev).clamp(0, 1)
    sharp_cells = None
    if band is not None:
        sharp_cells = (to_grid_nearest(np.asarray(band, np.float32), torch).to(dev) > 0.5)
        tn = to_grid_nearest(depth, torch).to(dev)
        t = torch.where(sharp_cells & torch.isfinite(tn), tn, t)
        known = torch.where(sharp_cells & torch.isfinite(tn), torch.ones_like(known), known)
    layers = means.shape[0] // (gsz * gsz)
    zf = means[: gsz * gsz, 2].reshape(gsz, gsz)
    k = torch.where((known > 0.5) & (zf > 1e-4) & torch.isfinite(t), t / zf.clamp(min=1e-4),
                    torch.ones_like(zf)).clamp(lo, hi)
    k = 1 + w * (k - 1)
    ks = [k.reshape(-1)]
    for layer in range(1, layers):
        kb = k
        if back is not None and sharp_cells is not None:
            tb = to_grid_nearest(back, torch).to(dev)
            zb = means[layer * gsz * gsz:(layer + 1) * gsz * gsz, 2].reshape(gsz, gsz)
            use = sharp_cells & torch.isfinite(tb) & (zb > 1e-4)
            kb = torch.where(use, (tb / zb.clamp(min=1e-4)).clamp(lo, hi), k)
        ks.append(kb.reshape(-1))
    kk = torch.cat(ks)[:, None]
    return means * kk, scales * kk, k


def align_affine(z_src, z_ref, iters=4, keep=0.8):
    """Robust (a, b) with a z_src + b ≈ z_ref (a > 0), trimmed least squares; scale only as a fallback."""
    import numpy as np

    x, y = np.asarray(z_src, np.float64), np.asarray(z_ref, np.float64)
    if len(x) < 20:
        return 1.0, 0.0
    a, b = float(np.median(y / x)), 0.0
    for _ in range(iters):
        r = np.abs(a * x + b - y)
        sel = r <= np.quantile(r, keep)
        A = np.stack([x[sel], np.ones(int(sel.sum()))], 1)
        a2, b2 = np.linalg.lstsq(A, y[sel], rcond=None)[0]
        if a2 <= 0:
            return float(np.median(y / x)), 0.0
        a, b = float(a2), float(b2)
    return a, b


def dilate(mask, r):
    """Binary dilation by r pixels (square neighbourhood), numpy only."""
    import numpy as np

    m = np.asarray(mask, bool)
    for _ in range(int(r)):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        m = m | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return m


def hole_depth(known, mono, hole, mono_ok, ring_px=12):
    """Composite depth for a filled view: the known (rendered) depth outside the holes, the monocular depth
    aligned to the known one's far side around the holes inside them. Returns (depth, valid, stats)."""
    import numpy as np

    known_ok = np.isfinite(known) & (known > 0)
    ring = dilate(hole, ring_px) & ~hole & known_ok & mono_ok
    stats = {"hole_fraction": round(float(hole.mean()), 5), "ring_pixels": int(ring.sum())}
    if ring.sum() >= 20:
        far = ring & (known >= np.median(known[ring]))      # the background the hole belongs to
        use = far if far.sum() >= 20 else ring
        a, b = align_affine(mono[use], known[use])
    else:     # no known surroundings: the monocular depth at the scene's own scale
        both = known_ok & mono_ok & ~hole
        a, b = (float(np.median(known[both] / mono[both])), 0.0) if both.sum() >= 20 else (1.0, 0.0)
    filled = a * mono + b
    depth = np.where(hole, filled, known)
    valid = np.where(hole, mono_ok & (filled > 0), known_ok)
    stats.update(scale=round(a, 5), shift=round(b, 5))
    return depth, valid, stats


def load_moge(path, version, device, torch):
    if version == "v3":
        os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "never")
        import flex_gemm

        flex_gemm.config.AUTOTUNE_MODE = "never"
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(path).to(device).eval()


def fill(req, torch, device):
    import math

    import numpy as np
    from PIL import Image
    from sharp.utils.gaussians import save_ply

    version = req.get("moge_version", "v2")
    progress(0.02, f"loading MoGe-{version[1:]} and SHARP")
    moge = load_moge(req["moge_path"], version, device, torch)
    predictor = load_predictor(req["checkpoint"], device, torch)
    stats = []
    views = req["views"]
    for i, v in enumerate(views):
        img = np.asarray(Image.open(v["image"]).convert("RGB"))
        h, w = img.shape[:2]
        hole = np.asarray(Image.open(v["mask"]).convert("L").resize((w, h), Image.NEAREST)) > 127
        with np.load(v["known"]) as z:
            known = z["depth"].astype(np.float64)
        t = torch.from_numpy(img).to(device).float().div(255).permute(2, 0, 1)
        kw = {"refine_steps": 3} if version == "v3" else {}
        fov = math.degrees(2 * math.atan(w / 2 / float(v["fx"])))
        with torch.no_grad():
            out = moge.infer(t, resolution_level=9, fov_x=fov, use_fp16=device.type == "cuda", **kw)
        mono = out["depth"].float().cpu().numpy().astype(np.float64)
        mono_ok = np.isfinite(mono) & (mono > 0)
        if "mask" in out:
            mono_ok &= out["mask"].cpu().numpy().astype(bool)
        mono = np.where(mono_ok, mono, 1.0)
        depth, valid, st = hole_depth(known, mono, hole, mono_ok, int(req.get("ring_px", 12)))
        g, info = predict(predictor, img, float(v["fx"]), device, torch, depth, valid)
        save_ply(g, float(v["fx"]), (h, w), v["out"])
        st.update(info)
        stats.append(st)
        log(f"fill view {i + 1}/{len(views)}: holes {st['hole_fraction']:.1%} of the picture, their depth tied to "
            f"{st['ring_pixels']:,} known pixels around them (×{st['scale']:.3f} {st['shift']:+.3f} m)")
        progress((i + 1) / len(views), f"new content for view {i + 1}/{len(views)}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    emit("result", outputs=[v["out"] for v in views], stats=stats, vram_peak_mib=vram_peak_mib(torch))


def people_normals(req, img, parts, device, torch):
    """Sapiens2's surface normals of the people (H, W, 3 at img's size, NaN outside their box), or None."""
    import human_geometry as hg

    if not (req.get("human_normal") and os.path.exists(req["human_normal"]) and parts is not None):
        return None
    progress(0.16, "people's fine relief (Sapiens2 normals)")
    net, _ = hg.load(req["human_normal"], "normal", device, torch)
    y0, y1, x0, x1 = parts["box"]
    n = hg.normal_map(net, img, parts["box"], torch, device)
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = np.full(img.shape[:2] + (3,), np.nan)
    out[y0:y1, x0:x1] = n
    return out


def relief_normals(depth, f_px, scene_n, people_n, people_w, cx=None, cy=None):
    """One normal map for the fine relief: the scene's normals (MoGe-3) and, inside the people, Sapiens2's;
    each first turned into the depth's own axes (precision.align_normals). Returns (normals or None, report)."""
    import precision as pr

    ref = pr.normals_from_depth(depth, f_px, cx, cy)
    rep, out = {}, None
    if scene_n is not None:
        a, st = pr.align_normals(scene_n, ref)
        rep["scene"] = {k: v for k, v in st.items() if k != "matrix"}
        if st["ok"]:
            out = a
    if people_n is not None and people_w is not None:
        inside = (people_w > 0.5) & np.isfinite(people_n).all(-1)
        a, st = pr.align_normals(people_n, ref, inside)
        rep["people"] = {k: v for k, v in st.items() if k != "matrix"}
        if st["ok"]:
            out = np.where(inside[..., None], a, out if out is not None else np.nan)
    return out, rep


def precise_depth(depth, img, f_px, normals, torch, device, cx=None, cy=None):
    """precision.integrate_normals then precision.resolve_edges on a depth map. Returns (depth, band, far,
    report)."""
    import precision as pr

    rep = {}
    if normals is not None:
        try:
            depth, rep["normals"] = pr.integrate_normals(depth, normals, f_px, torch, device=device, cx=cx, cy=cy)
        except Exception as e:  # noqa: BLE001 - an improvement, not a requirement
            rep["normals"] = {"used": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    else:
        rep["normals"] = {"used": False, "reason": "no normals that agree with the depth"}
    band = far = None
    try:
        depth, band, far, rep["edges"] = pr.resolve_edges(depth, img, torch, device=device)
    except Exception as e:  # noqa: BLE001
        rep["edges"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depth, band, far, rep


def relief_sheet(path, img, depths, f_px, labels, side=640):
    """relief.png: the photo, then each depth map shaded (light from the upper left) so its relief shows."""
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    size = (max(8, round(w * s)), max(8, round(h * s)))
    panels = [np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))]
    for d in depths:
        dd = np.asarray(Image.fromarray(np.nan_to_num(d, nan=0.0).astype(np.float32)).resize(size, Image.NEAREST),
                        np.float64)
        dd = np.where(dd > 0, dd, np.nan)
        panels.append(np.repeat(shade(dd, f_px * s)[..., None], 3, -1))
    sheet = Image.fromarray(np.concatenate(panels, 1).astype(np.uint8))
    dr = ImageDraw.Draw(sheet)
    for i, t in enumerate(["photo"] + list(labels)):
        dr.text((i * size[0] + 6, 4), t, fill=(255, 210, 90))
    sheet.save(path)
    return path


def detail_photo(req, predictor, img, depth, f_px, parts, device, torch, people_n=None):
    """Maximum detail (workers/detail_geometry.py): the photo at up to 3072 px (4608 for photos of 5000 px
    and more), MoGe-3 on crops fused into the whole-photo depth, SHARP on crops with every seen splat placed
    at that depth. With "precision" (rc7) the crops' normals and the people's (``people_n``, at img's size)
    add the fine relief and the depth edges are sent to one side first. Returns (gaussians, (W, H), focal
    length in those pixels, report, the photo at (W, H), the depth used)."""
    import torch.nn.functional as F
    from PIL import Image

    import detail_geometry as dg
    import precision as pr

    h, w = img.shape[:2]
    full = Image.open(req["photo"]).convert("RGB") if req.get("photo") else Image.fromarray(img)
    s = min(1.0, dg.work_side(max(full.size)) / max(full.size))
    W, H = max(round(full.width * s), w), max(round(full.height * s), h)
    work = np.asarray(full.resize((W, H), Image.LANCZOS)) if (W, H) != full.size else np.asarray(full)
    fw = f_px * W / w
    G = dg.resize_nan(depth, (H, W), torch)
    n = dg.tile_count(W, H)
    boxes = dg.tile_boxes(W, H, n) if n > 1 else [((0, 0, W, H), (0, 0, W, H))]
    rep = {"work_size": [W, H], "tiles": n * n}
    fused = G
    precise = bool(req.get("precision"))
    crop_normals = []
    if n > 1 and req.get("moge_path"):
        progress(0.45, f"fine relief: MoGe-3 on {n * n} parts of the photo at {W}x{H}")
        moge = load_moge(req["moge_path"], req.get("moge_version", "v3"), device, torch)
        crops = []
        for k, (crop, _) in enumerate(boxes):
            x0, y0, x1, y1 = crop
            c = work[y0:y1, x0:x1]
            cs = min(1.0, 1536 / max(c.shape[:2]))
            ci = c if cs >= 1 else np.asarray(Image.fromarray(c).resize((round(c.shape[1] * cs), round(c.shape[0] * cs)),
                                                                            Image.LANCZOS))
            t = torch.from_numpy(np.array(ci, copy=True)).to(device).float().div(255).permute(2, 0, 1)
            fov = float(np.degrees(2 * np.arctan((x1 - x0) / 2 / fw)))
            kw = {"refine_steps": 3} if req.get("moge_version", "v3") == "v3" else {}
            with torch.no_grad():
                o = moge.infer(t, resolution_level=9, fov_x=fov, use_fp16=device.type == "cuda", **kw)
            d = o["depth"].float()
            ok = torch.isfinite(d) & (d > 0)
            if "mask" in o:
                ok &= o["mask"].bool()
            d = torch.where(ok, d, torch.nan).cpu().numpy()
            crops.append((crop, dg.resize_nan(d, (y1 - y0, x1 - x0), torch)))
            if precise and o.get("normal") is not None:
                nm = o["normal"].float().cpu().numpy().astype(np.float64)
                crop_normals.append((crop, pr.resize_normals(nm, (y1 - y0, x1 - x0), torch)))
            progress(0.45 + 0.1 * (k + 1) / len(boxes), f"fine relief: part {k + 1}/{len(boxes)}")
        del moge
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fused, rep["depth_parts"] = dg.patch_fusion(G, crops, sigma=max(W, H) * 0.01, torch=torch)
    pw = None
    if parts is not None:                      # the people keep their own measured shape
        pw = F.interpolate(torch.as_tensor(parts["weight"], dtype=torch.float32)[None, None], size=(H, W),
                           mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float64)
        fused = np.where(np.isfinite(G), pw * G + (1 - pw) * fused, fused)
    band = far = None
    if precise:
        progress(0.555, "fine relief from the surface normals; clean depth edges")
        aligned = []
        for crop, nm in crop_normals:          # each part's normals into the photo's axes
            x0, y0, x1, y1 = crop
            ref = pr.normals_from_depth(fused[y0:y1, x0:x1], fw, cx=W / 2 - x0, cy=H / 2 - y0)
            a, st = pr.align_normals(nm, ref)
            if st["ok"]:
                aligned.append((crop, a))
        scene_n = pr.blend_normals((H, W), aligned) if aligned else None
        pn = pr.resize_normals(people_n, (H, W), torch) if people_n is not None else None
        normals, rep["normals_fit"] = relief_normals(fused, fw, scene_n, pn, pw)
        fused, band, far, rep["precision"] = precise_depth(fused, work, fw, normals, torch, device)
    weight = dg.edge_weight(fused, torch)
    if band is not None:
        weight = np.maximum(weight, band.astype(np.float64))
    if pw is not None:
        weight = np.maximum(weight, pw)
    rep["placed_share"] = round(float((weight > 0.5).mean()), 3)
    valid = np.isfinite(fused) & (fused > 0)
    keep_parts = []
    for k, (crop, core) in enumerate(boxes):
        progress(0.56 + 0.14 * k / len(boxes), f"SHARP on part {k + 1}/{len(boxes)} of the photo")
        x0, y0, x1, y1 = crop
        g, _ = predict(predictor, np.ascontiguousarray(work[y0:y1, x0:x1]), fw, device, torch,
                       fused[y0:y1, x0:x1], valid[y0:y1, x0:x1], cx=W / 2 - x0, cy=H / 2 - y0)
        with torch.no_grad():
            means, scales, _ = snap_to_depth(g, fused[y0:y1, x0:x1], weight[y0:y1, x0:x1], torch, lo=0.7, hi=1.4,
                                             band=None if band is None else band[y0:y1, x0:x1],
                                             back=None if far is None else far[y0:y1, x0:x1])
            own = torch.from_numpy(dg.owned(crop, core, SHARP_GRID).reshape(-1)).to(means.device)
            own = torch.cat([own] * (means.shape[0] // own.numel()))
            sel = own & (g.opacities[0] > 0.02)
            keep_parts.append([t.detach()[sel].cpu() for t in (means, scales, g.quaternions[0], g.colors[0],
                                                                g.opacities[0])])
        del g
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cat = [torch.cat([p[i] for p in keep_parts]) for i in range(5)]
    gd = type_of_gaussians()(*[c[None] for c in cat])
    rep["splats"] = int(cat[0].shape[0])
    log(f"maximum detail: {rep['splats']:,} splats from {len(boxes)} part(s) of the photo at {W}x{H}; "
        f"{rep['placed_share']:.0%} of it placed at the fine depth")
    return gd, (W, H), fw, rep, work, fused


def type_of_gaussians():
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D


def main(req):
    import numpy as np
    import torch

    env = torch_env(torch)
    if not env["cuda_available"] and not req.get("allow_cpu", False):
        emit("error", code="cuda_unavailable",
             message="CUDA is not available to PyTorch in this runtime (enable CPU mode in Settings to run slowly).")
        sys.exit(1)
    if not os.path.exists(req["checkpoint"]):
        emit("error", code="model_missing", message=f"SHARP checkpoint not found: {req['checkpoint']}")
        sys.exit(1)
    from sharp.utils.gaussians import save_ply

    device = torch.device("cuda" if env["cuda_available"] else "cpu")
    if req.get("mode") == "fill":
        fill(req, torch, device)
        return
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    geo = np.load(req["geometry"])
    img, points, mask, K = geo["image"], geo["points"].astype(np.float64), geo["mask"], geo["intrinsics"]
    h, w = img.shape[:2]
    f_px = float(K[0, 0])
    valid = mask & np.isfinite(points).all(-1) & (points[..., 2] > 0)
    log(f"the photo's lens: {np.degrees(2 * np.arctan(w / 2 / f_px)):.1f}° horizontal field of view; "
        f"MoGe depth on {valid.mean():.0%} of the pixels")

    # ---- people's own 3D shape (Sapiens2)
    people = parts = None
    if req.get("human_seg") and req.get("human_pointmap"):
        try:
            points, people, parts = people_geometry(req, img, points, valid, f_px, device, torch)
        except Exception as e:  # noqa: BLE001 - an improvement, not a requirement
            log(f"people's 3D shape unavailable ({type(e).__name__}: {str(e)[:300]}); the scene's depth is used")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- real-world scale: the distance between the eyes of the people, else the human model's distance
    faces, s_face = [], 1.0
    target = float(req.get("ipd_target_m") or IPD_TARGET_M)
    if req.get("face_model") and os.path.exists(req["face_model"]):
        try:
            faces = eye_distances(img, points, req["face_model"])
            prior = 1.0 / people["scene_vs_human_distance"] if people else None
            s_face = face_scale(faces, target, prior)
            if faces:
                log("eye distance of the faces found: " + ", ".join(
                    f"{f['ipd'] * 1000:.1f} mm" + (f" (left out: {f['rejected']})" if f.get("rejected") else "")
                    for f in faces) + (f"; scale x{s_face:.3f} to bring them to {target * 1000:.1f} mm"
                                       if any(f.get("used") for f in faces) else "; no usable face"))
            else:
                log("no face found for the scale check")
        except Exception as e:  # noqa: BLE001 - a check, not a requirement
            log(f"face scale check unavailable: {e}")
    scale, scale_from = 1.0, "measured depth"
    if any(f.get("used") for f in faces):
        scale, scale_from = s_face, "eye distance"
        if people:
            hs = 1.0 / people["scene_vs_human_distance"]
            log(f"cross-check: the human model would scale x{hs:.3f}")
    elif people:
        scale, scale_from = float(min(max(1.0 / people["scene_vs_human_distance"], 0.5), 2.0)), "human model"
        log(f"no face for the eye check: scale x{scale:.3f} from the human model's distance of the people")
    points = points * scale
    depth = np.where(valid, points[..., 2], np.nan)

    # ---- precision (rc7): fine relief from the surface normals, clean depth edges
    precise = bool(req.get("precision"))
    prec, band, far, people_n, before_relief = {}, None, None, None, depth.copy()
    if precise:
        if parts is not None:
            try:
                people_n = people_normals(req, img, parts, device, torch)
            except Exception as e:  # noqa: BLE001 - an improvement, not a requirement
                log(f"people's normals unavailable ({type(e).__name__}: {str(e)[:200]})")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        scene_n = None
        if "normal" in geo.files:
            scene_n = geo["normal"].astype(np.float64)
            scene_n = np.where((np.linalg.norm(scene_n, axis=-1) > 0.5)[..., None], scene_n, np.nan)
        progress(0.18, "fine relief from the surface normals; clean depth edges")
        normals, prec["normals_fit"] = relief_normals(depth, f_px, scene_n, people_n,
                                                      parts["weight"] if parts is not None else None)
        depth, band, far, more = precise_depth(depth, img, f_px, normals, torch, device)
        prec.update(more)
        valid = np.isfinite(depth) & (depth > 0)
        nr = prec.get("normals") or {}
        if nr.get("used"):
            log(f"fine relief from the surface normals: {nr['rms_change'] * 100:.2f} % depth change (rms), finer "
                f"than {nr['cutoff_px']:.0f} px" + (" (people: Sapiens2)" if people_n is not None else ""))
        else:
            log(f"no fine relief from normals: {nr.get('reason') or nr.get('error')}")
        if (prec.get("edges") or {}).get("band_share"):
            log(f"depth edges sent to one side on {prec['edges']['band_share']:.1%} of the picture")

    progress(0.2, "loading SHARP")
    predictor = load_predictor(req["checkpoint"], device, torch)
    progress(0.4, "SHARP with the photo's measured depth")
    g, info = predict(predictor, img, f_px, device, torch, depth, valid)
    if parts is not None:
        sheet_depths = [parts["scene_z"] * scale, depth]
    if precise:
        # every splat the photo sees at the measured depth (people, fine relief and edges included)
        import detail_geometry as dg

        wgt = dg.edge_weight(depth, torch)
        if band is not None:
            wgt = np.maximum(wgt, band.astype(np.float64))
        if parts is not None:
            wgt = np.maximum(wgt, parts["weight"])
        with torch.no_grad():
            means, scales, k = snap_to_depth(g, depth, wgt, torch, lo=0.7, hi=1.4, band=band, back=far)
            g = g._replace(mean_vectors=means[None], singular_values=scales[None])
        prec["placed_share"] = round(float((wgt > 0.5).mean()), 3)
        if people:
            people["snapped"] = True
        log(f"{prec['placed_share']:.0%} of the photo's splats placed at the measured depth")
    elif parts is not None and req.get("snap_people", True):
        # SHARP follows the injected depth only through a smooth learned alignment: put the people's
        # splats exactly at their measured 3D shape (MoGe-3 + Sapiens2), as for splat videos
        with torch.no_grad():
            means, scales, k = snap_to_depth(g, depth, parts["weight"], torch)
            g = g._replace(mean_vectors=means[None], singular_values=scales[None])
        inside = k[k != 1]
        moved = float((inside - 1).abs().quantile(0.9)) if inside.numel() else 0.0
        people["snapped"] = True
        people["snap_change_p90"] = round(moved, 4)
        log(f"people's splats placed at their measured 3D shape (90 % moved by up to {moved * 100:.1f} %)")
    if parts is not None:
        with torch.no_grad():
            sheet_depths.append(front_depth(g.mean_vectors[0], h, w, torch))
        try:
            people["sheet"] = people_sheet(os.path.join(out_dir, "people_shape.png"), img, parts, sheet_depths, f_px)
            log("people_shape.png: the picture | scene depth | with the people's own shape | the splats as made")
        except Exception as e:  # noqa: BLE001 - a diagnostic
            log(f"people shape picture skipped: {e}")
    locked = os.path.join(out_dir, "locked.ply")
    save_ply(g, f_px, (h, w), locked)
    outputs = {"locked": locked}
    size_out, f_out, detail, final_depth = [w, h], f_px, None, depth
    if req.get("detail"):
        try:
            gd, (W, H), fw, detail, _work, fused = detail_photo(req, predictor, img, depth, f_px, parts, device, torch,
                                                               people_n)
            outputs["standard"] = locked                       # the normal result, for comparison
            locked = outputs["locked"] = os.path.join(out_dir, "locked_detail.ply")
            save_ply(gd, fw, (H, W), locked)
            size_out, f_out, final_depth = [W, H], fw, fused
            del gd
        except Exception as e:  # noqa: BLE001 - fall back to the normal result
            if device.type == "cuda":
                torch.cuda.empty_cache()
            detail = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            log(f"maximum detail failed ({detail['error']}); the normal result is used")
    log(f"SHARP's own depth was {info['sharp_own_scale']:.3f}× the measured one; aligned to it")
    if req.get("plain", True):
        progress(0.75, "SHARP alone (for comparison)")
        g2, _ = predict(predictor, img, f_px, device, torch)
        outputs["plain"] = os.path.join(out_dir, "plain.ply")
        save_ply(g2, f_px, (h, w), outputs["plain"])
        del g2
    del predictor, g
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # the fit to the photo (workers/refine.py) runs afterwards in the renderer runtime (rc8: gsplat needs its
    # prebuilt CUDA kernels, which this runtime's torch has none of)
    if precise:
        try:
            panels, labels = [], []
            if "depth_moge" in geo.files:
                panels.append(np.where(geo["depth_moge"] > 0, geo["depth_moge"], np.nan))
                labels.append("MoGe-3")
            if "depth_pro" in geo.files:
                panels.append(np.where(mask, geo["points"][..., 2].astype(np.float64), np.nan))
                labels.append("+ Depth Pro")
            panels.append(before_relief)
            labels.append("+ people (Sapiens2)" if parts is not None else "depth used before")
            panels.append(final_depth)
            labels.append("final: + normals, clean edges" + (" (max detail)" if final_depth is not depth else ""))
            fd = final_depth
            if fd.shape != depth.shape:              # the detail depth, shown at the photo's size
                from detail_geometry import resize_nan

                panels[-1] = resize_nan(fd, depth.shape, torch)
            outputs["relief"] = relief_sheet(os.path.join(out_dir, "relief.png"), img, panels, f_px, labels)
            log("relief.png: the photo | each depth step shaded, so its relief shows")
        except Exception as e:  # noqa: BLE001 - a diagnostic
            log(f"relief picture skipped: {e}")
    progress(1.0, "done")
    emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), f_px=f_out, size=size_out, detail=detail,
         faces=faces, face_scale=s_face, scale=scale, scale_from=scale_from, ipd_target_m=target, people=people,
         precision=prec if precise else None, **info)


if __name__ == "__main__":
    run(main)
