"""Apple SHARP with an injected depth (runtime 'photo-cu128') - Real 3D, "the photo locked" (rc56).

SHARP's own depth is monocular with a guessed lens (30 mm without EXIF), so its scale and volume are
arbitrary. Its network can align its depth to a given one: ``RGBGaussianPredictor.forward(image,
disparity_factor, depth=...)`` runs a learned UNet (``LearnedAlignment``, part of the released checkpoint)
that turns the given depth into a smooth local scale map for SHARP's two depth layers. The CLI never passes
it; this worker does, with MoGe's metric depth of the photo at the photo's real lens.

Request (mode "photo"):
  {"mode": "photo", "checkpoint": ".../sharp_2572gikvuh.pt", "geometry": ".../photo.npz" (moge_worker output:
   points, mask, intrinsics, image), "output_dir": "...", "face_model": ".../yunet.onnx" | null,
   "ipd_range_m": [0.056, 0.070], "plain": true, "allow_cpu": false}
Writes output_dir/locked.ply (SHARP PLY with metadata) and, with "plain", output_dir/plain.ply (SHARP alone,
for comparison). Result: scale diagnostics (how far SHARP's own depth was from MoGe's, the face check).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

INTERNAL = 1536               # SHARP's working resolution (square)
IPD_RANGE_M = (0.056, 0.070)  # adult interpupillary distance, ~5th-95th percentile


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


def face_scale(faces, ipd_range=IPD_RANGE_M):
    """Factor that brings the median eye distance into the adult range (1.0 when already inside it)."""
    import numpy as np

    if not faces:
        return 1.0
    ipd = float(np.median([f["ipd"] for f in faces]))
    if ipd <= 0:
        return 1.0
    target = min(max(ipd, ipd_range[0]), ipd_range[1])
    return float(min(max(target / ipd, 0.5), 2.0))


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
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    geo = np.load(req["geometry"])
    img, points, mask, K = geo["image"], geo["points"].astype(np.float64), geo["mask"], geo["intrinsics"]
    h, w = img.shape[:2]
    f_px = float(K[0, 0])
    valid = mask & np.isfinite(points).all(-1) & (points[..., 2] > 0)
    log(f"the photo's lens: {np.degrees(2 * np.arctan(w / 2 / f_px)):.1f}° horizontal field of view; "
        f"MoGe depth on {valid.mean():.0%} of the pixels")

    # ---- real-world scale check: the distance between the eyes of the people in the photo
    faces, s_face = [], 1.0
    if req.get("face_model") and os.path.exists(req["face_model"]):
        try:
            faces = eye_distances(img, points, req["face_model"])
            s_face = face_scale(faces, tuple(req.get("ipd_range_m") or IPD_RANGE_M))
            if faces:
                log("eye distance of the faces found: " + ", ".join(f"{f['ipd'] * 1000:.0f} mm" for f in faces) +
                    (f"; scale corrected by {s_face:.3f}" if s_face != 1.0 else "; scale confirmed"))
            else:
                log("no face found for the scale check")
        except Exception as e:  # noqa: BLE001 - a check, not a requirement
            log(f"face scale check unavailable: {e}")
    points = points * s_face
    depth = np.where(valid, points[..., 2], np.nan)

    progress(0.2, "loading SHARP")
    predictor = load_predictor(req["checkpoint"], device, torch)
    progress(0.4, "SHARP with the photo's measured depth")
    g, info = predict(predictor, img, f_px, device, torch, depth, valid)
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
         faces=faces, face_scale=s_face, **info)


if __name__ == "__main__":
    run(main)
