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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

INTERNAL = 1536               # SHARP's working resolution (square)
IPD_TARGET_M = 0.063          # adult interpupillary distance, population median (women ~62, men ~64 mm)


def eye_distances(img, points, model_path, min_face_px=40):
    """3D distance between the eyes of every face found (YuNet, OpenCV), in the units of ``points``."""
    import cv2
    import numpy as np

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
            eyes.append(np.median(win, 0))
        if len(eyes) == 2:
            out.append({"box": [float(v) for v in f[:4]], "score": float(f[14]),
                        "ipd": float(np.linalg.norm(eyes[0] - eyes[1]))})
    return out


def face_scale(faces, target=IPD_TARGET_M):
    """Factor that brings the median eye distance of the faces to ``target`` metres (rc58: always, not only
    outside a range - owner: people always came out too big). 1.0 without faces; never more than 2x."""
    import numpy as np

    if not faces:
        return 1.0
    ipd = float(np.median([f["ipd"] for f in faces]))
    if ipd <= 0:
        return 1.0
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


def predict(predictor, img, f_px, device, torch, depth=None, valid=None):
    """SHARP on an image (H,W,3 uint8) with focal length f_px; ``depth`` (H,W metric, ``valid`` mask) is
    injected when given. Returns (gaussians in metric camera space, diagnostics)."""
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
        K = torch.tensor([[f_px, 0, w / 2, 0], [0, f_px, h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().to(device)
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


def snap_to_depth(g, depth, weight, torch, lo=0.6, hi=1.6):
    """Put SHARP's splats exactly at ``depth`` (H, W metres, NaN = unknown) where ``weight`` (H, W in [0, 1])
    says so, blending to SHARP's own placement where it falls to 0.

    SHARP only follows an injected depth through its learned alignment, which is smooth: the fine relief of
    a person (nose, face, folds of clothes) from the human model is mostly lost, more so on soft video frames
    whose own SHARP depth is flat. Each pixel's front splat is moved along its line of sight to the given
    depth, and the splat behind it by the same factor (sizes too, so the picture from the camera does not
    change). Returns (means, scales) for all 2 x 768 x 768 splats."""
    gsz = SHARP_GRID
    means, scales = g.mean_vectors[0], g.singular_values[0]
    dev = means.device
    t, known = to_grid(depth, torch)
    w, _ = to_grid(weight, torch)
    t, known, w = t.to(dev), known.to(dev), w.to(dev).clamp(0, 1)
    zf = means[: gsz * gsz, 2].reshape(gsz, gsz)
    k = torch.where((known > 0.5) & (zf > 1e-4) & torch.isfinite(t), t / zf.clamp(min=1e-4),
                    torch.ones_like(zf)).clamp(lo, hi)
    k = 1 + w * (k - 1)
    kk = torch.cat([k.reshape(-1)] * (means.shape[0] // (gsz * gsz)))[:, None]
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
            s_face = face_scale(faces, target)
            if faces:
                log("eye distance of the faces found: " + ", ".join(f"{f['ipd'] * 1000:.1f} mm" for f in faces) +
                    f"; scale x{s_face:.3f} to bring them to {target * 1000:.1f} mm")
            else:
                log("no face found for the scale check")
        except Exception as e:  # noqa: BLE001 - a check, not a requirement
            log(f"face scale check unavailable: {e}")
    scale, scale_from = 1.0, "measured depth"
    if faces:
        scale, scale_from = s_face, "eye distance"
        if people:
            hs = 1.0 / people["scene_vs_human_distance"]
            log(f"cross-check: the human model would scale x{hs:.3f}")
    elif people:
        scale, scale_from = float(min(max(1.0 / people["scene_vs_human_distance"], 0.5), 2.0)), "human model"
        log(f"no face for the eye check: scale x{scale:.3f} from the human model's distance of the people")
    points = points * scale
    depth = np.where(valid, points[..., 2], np.nan)

    progress(0.2, "loading SHARP")
    predictor = load_predictor(req["checkpoint"], device, torch)
    progress(0.4, "SHARP with the photo's measured depth")
    g, info = predict(predictor, img, f_px, device, torch, depth, valid)
    if parts is not None:
        sheet_depths = [parts["scene_z"] * scale, depth]
        if req.get("snap_people", True):
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
    log(f"SHARP's own depth was {info['sharp_own_scale']:.3f}× the measured one; aligned to it")
    if req.get("plain", True):
        progress(0.75, "SHARP alone (for comparison)")
        g2, _ = predict(predictor, img, f_px, device, torch)
        outputs["plain"] = os.path.join(out_dir, "plain.ply")
        save_ply(g2, f_px, (h, w), outputs["plain"])
    progress(1.0, "done")
    emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), f_px=f_px, size=[w, h],
         faces=faces, face_scale=s_face, scale=scale, scale_from=scale_from, ipd_target_m=target, people=people,
         **info)


if __name__ == "__main__":
    run(main)
