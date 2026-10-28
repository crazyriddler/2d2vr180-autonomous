"""MoGe-2 / MoGe-3 monocular metric geometry worker (runs in runtime 'photo-cu128').

Request:
  {"model_path": ".../model.pt", "images": [{"path": "...", "out": ".../frame.npz", "fov_x": 67.4}],
   "max_side": 1536, "fp16": true, "resolution_level": 9, "version": "v2" | "v3", "refine_steps": 3,
   "precision": {"passes": true, "depth_pro": ".../depth_pro.pt" | null, "lens_check": true}}
``fov_x`` (optional, degrees): the photo's real horizontal field of view (EXIF), which fixes the depth/size
ratio of the whole result. MoGe-3 (July 2026) refines MoGe-2's estimate in 3D (sparse volumetric
refinement through FlexGEMM, run by Triton on Windows) for finer, more accurate geometry.
Writes per image an .npz with points (H,W,3 float32, OpenCV camera frame, metres),
mask (H,W bool), intrinsics (3x3, pixels), image (H,W,3 uint8) and, when the model has them, normal
(H,W,3 float16, OpenCV camera frame).

"precision" (2.0.0-rc7, photos; workers/precision.py): the EXIF lens checked against MoGe's and Depth Pro's
own estimates; MoGe run on the photo, its mirror image and at a coarser level, median depth; Depth Pro's
depth joined to it (shape from both, edges from Depth Pro). The per-model depths are kept in the .npz
(depth_moge, depth_pro) for the relief diagnostics.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402


def moge_pass(model, arr, fov_x, level, flip, version, fp16, refine_steps, device, torch):
    """One MoGe pass -> (depth (H,W, NaN = unknown), normal (H,W,3) or None, mask, intrinsics in pixels)."""
    import numpy as np

    a = arr[:, ::-1] if flip else arr
    t = torch.from_numpy(np.array(a, copy=True)).to(device).float().div(255).permute(2, 0, 1)
    kw = {"refine_steps": int(refine_steps)} if version == "v3" else {}
    with torch.no_grad():
        out = model.infer(t, resolution_level=int(level), fov_x=fov_x, use_fp16=fp16 and device.type == "cuda", **kw)
    h, w = arr.shape[:2]
    K = out["intrinsics"].float().cpu().numpy().copy()
    K[0] *= w  # MoGe returns normalised intrinsics
    K[1] *= h
    depth = out["depth"].float().cpu().numpy().astype(np.float64)
    mask = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.ones((h, w), bool)
    normal = out["normal"].float().cpu().numpy().astype(np.float64) if out.get("normal") is not None else None
    if flip:
        depth, mask = depth[:, ::-1], mask[:, ::-1]
        if normal is not None:
            normal = normal[:, ::-1] * np.array([-1.0, 1.0, 1.0])       # mirrored: x points the other way
    depth = np.where(mask & np.isfinite(depth) & (depth > 0), depth, np.nan)
    return depth, normal, mask, K


def fov_of(K, w):
    import numpy as np

    return float(np.degrees(2 * np.arctan(w / 2 / K[0, 0])))


def load_depth_pro(path, device, torch):
    import dataclasses

    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=path)
    half = device.type == "cuda"
    model, _ = depth_pro.create_model_and_transforms(cfg, device=device,
                                                     precision=torch.half if half else torch.float32)
    return model.eval()


def depth_pro_run(model, arr, device, torch):
    """Depth Pro on an image: (canonical inverse depth at the image's size, its own horizontal fov in
    degrees). Depth for a lens of f pixels: f / (W * canonical)."""
    import numpy as np
    import torch.nn.functional as F

    h, w = arr.shape[:2]
    dtype = next(model.parameters()).dtype
    x = torch.from_numpy(np.array(arr, copy=True)).to(device).float().div(255).permute(2, 0, 1)[None]
    x = ((x - 0.5) / 0.5).to(dtype)
    x = F.interpolate(x, size=(model.img_size, model.img_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        canon, fov = model(x)
    canon = F.interpolate(canon.float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return canon.cpu().numpy().astype(np.float64), float(fov.float().reshape(-1)[0]) if fov is not None else None


def precise_geometry(model, arr, item, req, version, device, torch):
    """The precision path of one photo -> (depth, normal or None, mask, K, report)."""
    import numpy as np

    import precision as pr

    prec = req["precision"]
    h, w = arr.shape[:2]
    fp16 = bool(req.get("fp16", True))
    level = int(req.get("resolution_level", 9))
    steps = int(req.get("refine_steps", 3))
    rep = {}
    exif = item.get("fov_x")

    # the lens: the EXIF one checked against the models' own estimates
    d0, n0, m0, K0 = moge_pass(model, arr, None, level, False, version, fp16, steps, device, torch)
    est = {"MoGe": fov_of(K0, w)}
    dp, canon, dp_fov = None, None, None
    if prec.get("depth_pro") and os.path.exists(prec["depth_pro"]):
        try:
            progress(0.25, "Depth Pro: second opinion on the depth and the lens")
            dp = load_depth_pro(prec["depth_pro"], device, torch)
            canon, dp_fov = depth_pro_run(dp, arr, device, torch)
            est["Depth Pro"] = dp_fov
            del dp
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - an improvement, not a requirement
            rep["depth_pro_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"Depth Pro unavailable ({rep['depth_pro_error']}); MoGe alone is used")
            canon = None
    if prec.get("lens_check", True):
        fov, src, note = pr.lens_choice(exif, est)
    else:
        fov, src, note = exif, "EXIF" if exif else "model", "lens check off"
    rep["lens"] = {"exif_deg": exif, "estimates_deg": {k: round(v, 2) for k, v in est.items() if v},
                   "used_deg": round(fov, 2) if fov else round(est["MoGe"], 2), "from": src, "note": note}
    log("the photo's lens: " + note)

    # MoGe on the photo, its mirror image and at a coarser level, at that lens
    passes = [(level, False)] + ([(level, True), (max(0, level - 2), False)] if prec.get("passes", True) else [])
    depths, normals, K, mask = [], [], None, None
    for k, (lv, flip) in enumerate(passes):
        progress(0.35 + 0.45 * k / len(passes), f"MoGe pass {k + 1}/{len(passes)}")
        if k == 0 and not fov and not flip:
            d, n, m, Kp = d0, n0, m0, K0           # no fixed lens: the first estimate is the first pass
        else:
            d, n, m, Kp = moge_pass(model, arr, fov or est["MoGe"], lv, flip, version, fp16, steps, device, torch)
        depths.append(d)
        if n is not None:
            normals.append(n)
        if k == 0:
            K, mask = Kp, m
    depth, spread = pr.combine_passes(depths)
    rep["passes"] = len(passes)
    if len(passes) > 1:
        rep["pass_spread_median"] = round(float(np.nanmedian(spread)), 5)
    normal = None
    if normals:
        acc = np.nansum(np.stack(normals), 0)
        normal = acc / np.maximum(np.linalg.norm(acc, axis=-1, keepdims=True), 1e-9)
    moge_depth = depth.copy()

    # Depth Pro joined to it
    dpro = None
    if canon is not None:
        dpro = np.where(np.isfinite(canon) & (canon > 1e-6), K[0, 0] / (w * np.maximum(canon, 1e-6)), np.nan)
        joined, st = pr.depth_ensemble(depth, dpro, sigma=0.01 * max(h, w) + 2, torch=torch)
        rep["depth_pro"] = st
        if st.get("used"):
            depth = np.where(mask, joined, np.nan)
            log(f"Depth Pro joined: overall shape agrees within {st['disagreement_median'] * 100:.1f} % "
                f"(median), its edges used on {st['edge_share']:.0%} of the picture")
        else:
            log(f"Depth Pro not used: {st.get('reason')}")
    return depth, normal, mask, K, rep, moge_depth, dpro


def main(req):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    import precision as pr

    env = torch_env(torch)
    if not env["cuda_available"] and not req.get("allow_cpu", False):
        emit("error", code="cuda_unavailable",
             message="CUDA is not available to PyTorch in this runtime (driver too old or no NVIDIA GPU).")
        sys.exit(1)
    device = torch.device("cuda" if env["cuda_available"] else "cpu")
    version = req.get("version", "v2")
    if version == "v3":
        os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "never")
        import flex_gemm

        flex_gemm.config.AUTOTUNE_MODE = "never"     # as MoGe's own app: no slow first-call autotuning
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel

    progress(0.02, f"loading MoGe-{version[1:]}")
    model = MoGeModel.from_pretrained(req["model_path"]).to(device).eval()
    images = req["images"]
    max_side = int(req.get("max_side", 1536))
    reports = []
    for i, item in enumerate(images):
        with Image.open(item["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                s = max_side / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)
        h, w = arr.shape[:2]
        extra, rep = {}, {}
        if req.get("precision"):
            depth, normal, mask, K, rep, dm, dpro = precise_geometry(model, arr, item, req, version, device, torch)
            extra["depth_moge"] = np.nan_to_num(dm, nan=0.0).astype(np.float32)
            if dpro is not None:
                extra["depth_pro"] = np.nan_to_num(dpro, nan=0.0).astype(np.float32)
        else:
            depth, normal, mask, K = moge_pass(model, arr, item.get("fov_x"), int(req.get("resolution_level", 9)),
                                               False, version, bool(req.get("fp16", True)),
                                               int(req.get("refine_steps", 3)), device, torch)
        points = pr.rays(h, w, K[0, 0], K[1, 1], K[0, 2], K[1, 2]) * np.nan_to_num(depth, nan=0.0)[..., None]
        if normal is not None:
            extra["normal"] = np.nan_to_num(normal, nan=0.0).astype(np.float16)
        np.savez_compressed(
            item["out"],
            points=points.astype(np.float32),
            mask=mask & np.isfinite(depth),
            intrinsics=K.astype(np.float64),
            image=arr,
            **extra,
        )
        reports.append(rep)
        progress((i + 1) / len(images), f"geometry {i + 1}/{len(images)}")
        if device.type == "cuda" and (i % 8 == 7):
            torch.cuda.empty_cache()
    emit("result", outputs=[it["out"] for it in images], vram_peak_mib=vram_peak_mib(torch),
         model=f"moge-{version[1:]}", metric=True, precision=reports)


if __name__ == "__main__":
    run(main)
