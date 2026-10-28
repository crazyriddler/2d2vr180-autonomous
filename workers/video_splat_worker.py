"""Splat video: every frame of a video made with the Real 3D technique (runtime 'photo-cu128', 2D2VR 2.0).

Per frame, exactly as for a photo (workers/sharp_depth_worker.py): MoGe-3's measured depth at the video's
lens, people's shape from Sapiens2, the real-world scale from the eyes, and Apple SHARP aligned to that
depth. What a video adds:
  * one lens and one real-world scale for the whole clip, measured on a few frames spread over it
    (calibration), so the scene does not breathe from frame to frame;
  * depth stabilisation: each frame's depth is matched to the previous frame's on the pixels that did not
    change, which removes the small scale jumps monocular depth makes between frames;
  * a fixed subset of SHARP's pixel-aligned splats, the same in every frame, so still parts of the video
    are identical from frame to frame and compress to nothing (splv_codec.py);
  * every model is loaded once; frames are written as they are done (frames/NNNNNN.bin), so memory stays
    flat and an interrupted run resumes where it stopped.

Request:
  {"frames": ["frame_000000.png", ...], "output_dir": "...", "checkpoint": ".../sharp_2572gikvuh.pt",
   "moge_path": ".../model.pt", "face_model": ... | null, "human_seg": ... | null, "human_pointmap": ... | null,
   "ipd_target_m": 0.063, "fov_x": null | degrees, "grid": [1.5, 3] (SHARP grid steps: front layer, back layer),
   "keyframe_interval": 15, "moge_max_side": 1024, "people": true, "allow_cpu": false}
Result: {"outputs": {"frames_dir", "meta"}, "frames", "count", "camera", "quant", "scale", "scale_from", ...}
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import cap_vram, emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

SHARP_GRID = 768          # SHARP: 2 layers of 768 x 768 splats (stride 2 on its 1536 x 1536 input)


def grid_index(step_front: float, step_back: float, n_layers: int = 2, g: int = SHARP_GRID):
    """Indices of the splats kept from SHARP's flattened [layer, row, col] grid, and each one's step (a
    fractional step keeps e.g. 2 of every 3 rows and columns)."""
    import numpy as np

    idx, step = [], []
    for layer, s in zip(range(n_layers), (step_front, step_back)):
        s = max(float(s), 1.0)
        lines = np.unique(np.minimum(np.round(np.arange(0, g, s)), g - 1).astype(np.int64))
        ys, xs = np.meshgrid(lines, lines, indexing="ij")
        idx.append((layer * g * g + ys * g + xs).ravel())
        step.append(np.full(ys.size, g / len(lines), np.float32))
    return np.concatenate(idx), np.concatenate(step)


def stabilise(depth, prev_depth, img, prev_img, valid, tol=12):
    """Factor bringing this frame's depth to the previous frame's on the pixels that did not change (colour
    within ``tol`` levels): monocular depth jumps a few percent between frames. None when too few pixels
    (first frame, a cut, a fast pan) or when the jump is too large to be the depth's own wobble."""
    import numpy as np

    if prev_depth is None or prev_depth.shape != depth.shape:
        return None, 0.0
    still = valid & (np.abs(img.astype(np.int16) - prev_img.astype(np.int16)).max(-1) <= tol)
    still &= np.isfinite(prev_depth) & (prev_depth > 0) & np.isfinite(depth) & (depth > 0)
    share = float(still.mean())
    if still.sum() < 0.05 * still.size:
        return None, share
    r = float(np.median(prev_depth[still] / depth[still]))
    return (r if 0.7 < r < 1.4 else None), share


def main(req):
    import numpy as np
    import torch
    from PIL import Image

    import human_geometry as hg
    import sharp_depth_worker as sd
    import splv_codec as codec

    env = torch_env(torch)
    if not env["cuda_available"] and not req.get("allow_cpu", False):
        emit("error", code="cuda_unavailable", message="Splat videos need an NVIDIA GPU (CUDA).")
        sys.exit(1)
    device = torch.device("cuda" if env["cuda_available"] else "cpu")
    if device.type == "cuda":
        cap_vram(torch, 0.92)
    frames = req["frames"]
    out = req["output_dir"]
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)
    meta_path = os.path.join(out, "video_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    def load(i):
        return np.asarray(Image.open(frames[i]).convert("RGB"))

    # ---- models (once)
    progress(0.0, "loading MoGe-3, SHARP and the people models")
    moge = sd.load_moge(req["moge_path"], "v3", device, torch)
    predictor = sd.load_predictor(req["checkpoint"], device, torch)
    seg = pm = None
    if req.get("people", True) and req.get("human_seg") and req.get("human_pointmap"):
        try:
            seg, _ = hg.load(req["human_seg"], "seg", device, torch)
            pm, arch = hg.load(req["human_pointmap"], "pointmap", device, torch)
            log(f"people's shape from Sapiens2 ({arch}) in every frame")
        except Exception as e:  # noqa: BLE001 - an improvement, not a requirement
            seg = pm = None
            log(f"people's shape unavailable ({type(e).__name__}: {str(e)[:200]}); the scene's depth is used")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    max_side = int(req.get("moge_max_side", 1024))

    def geometry(img, fov):
        """MoGe-3 on the frame (at most max_side px), its depth brought back to the frame's full size.
        Returns (depth (H, W) metres with NaN where unknown, the lens in full-size pixels, the field of view)."""
        import torch.nn.functional as F

        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        small = img if s >= 1 else np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR))
        t = torch.from_numpy(np.ascontiguousarray(small).copy()).to(device).float().div(255).permute(2, 0, 1)
        with torch.no_grad():
            o = moge.infer(t, resolution_level=9, fov_x=fov, use_fp16=device.type == "cuda", refine_steps=2)
            z = o["points"][..., 2].float()
            ok = torch.isfinite(z) & (z > 0)
            if "mask" in o:
                ok &= o["mask"].bool()
            if z.shape != (h, w):                       # normalised bilinear: unknown pixels do not leak in
                num = F.interpolate(torch.where(ok, z, 0)[None, None], size=(h, w), mode="bilinear",
                                    align_corners=False)[0, 0]
                den = F.interpolate(ok.float()[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
                z, ok = num / den.clamp(min=1e-6), den > 0.99
            depth = torch.where(ok, z, torch.nan).cpu().numpy().astype(np.float64)
        K = o["intrinsics"].float().cpu().numpy().astype(np.float64)
        f_px = float(K[0, 0]) * w                        # normalised intrinsics → full-size pixels
        return depth, f_px, float(np.degrees(2 * np.arctan(0.5 / K[0, 0])))

    def unproject(depth, f_px):
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w]
        return np.stack([(xs + 0.5 - w / 2) / f_px * depth, (ys + 0.5 - h / 2) / f_px * depth, depth], -1)

    def people(img, depth, f_px):
        """The people's own 3D shape (Sapiens2, on the full frame) put into the depth. Returns (depth, ratio of
        the scene's distance of the people to the human model's, weight map: 1 inside the people, 0 outside)."""
        none = (depth, None, None)
        if seg is None:
            return none
        mask = hg.people_mask(seg, img, torch, device)
        box = hg.crop_box(mask)
        if box is None:
            return none
        d_crop, _ = hg.pointmap_depth(pm, img, box, f_px, torch, device)
        y0, y1, x0, x1 = box
        human = np.full(img.shape[:2], np.nan)
        human[y0:y1, x0:x1] = d_crop
        valid = np.isfinite(depth) & (depth > 0)
        fused, ratio = hg.fuse_people(np.where(valid, depth, np.nan), valid, human, mask)
        if ratio is None:
            return none
        both = valid & mask & np.isfinite(human) & (human > 0)
        return np.where(valid, fused, np.nan), ratio, hg.inner_weight(both, ramp=6)

    # ---- calibration: one lens and one real-world scale for the clip
    if "fov_x" not in meta:
        sample = sorted({int(round(v)) for v in np.linspace(0, len(frames) - 1, min(8, len(frames)))})
        progress(0.02, f"measuring the lens and the real-world scale on {len(sample)} frames")
        fovs = [geometry(load(i), req.get("fov_x"))[2] for i in sample]
        fov = float(req.get("fov_x") or np.median(fovs))
        eyes, humans = [], []
        target = float(req.get("ipd_target_m") or sd.IPD_TARGET_M)
        for i in sample:
            img = load(i)
            depth, f_px, _ = geometry(img, fov)
            depth, ratio, _ = people(img, depth, f_px)
            if ratio:
                humans.append(1.0 / ratio)
            if req.get("face_model") and os.path.exists(req["face_model"]):
                try:
                    faces = sd.eye_distances(img, unproject(depth, f_px), req["face_model"])
                    s_eyes = sd.face_scale(faces, target, 1.0 / ratio if ratio else None)
                    if any(f.get("used") for f in faces):
                        eyes.append(s_eyes)
                except Exception as e:  # noqa: BLE001
                    log(f"face check unavailable: {e}")
        if eyes:
            scale, scale_from = float(np.median(eyes)), "eye distance"
        elif humans:
            scale, scale_from = float(min(max(np.median(humans), 0.5), 2.0)), "human model"
        else:
            scale, scale_from = 1.0, "measured depth"
        meta = {"fov_x": fov, "scale": scale, "scale_from": scale_from, "eyes_frames": len(eyes),
                "lens_samples": fovs, "ipd_target_m": target}
        json.dump(meta, open(meta_path, "w"))
        log(f"clip lens {fov:.1f}° horizontal; real-world scale ×{scale:.3f} from the {scale_from}"
            + (f" ({len(eyes)} of {len(sample)} frames with faces)" if eyes else ""))
    fov, scale = meta["fov_x"], meta["scale"]
    snap = bool(req.get("snap_people", True))
    sheet_done = os.path.exists(os.path.join(out, "people_shape.png"))

    def people_sheet(img, scene_depth, fused_depth, means, g, weight, f_px, k):
        """people_shape.png (first frame with people): the picture, the scene's depth, with the people's own
        shape, and the splats' depth as made - shaded so the relief shows."""
        try:
            import torch.nn.functional as F

            gsz = sd.SHARP_GRID
            h, w = img.shape[:2]
            zf = means[: gsz * gsz, 2].reshape(1, 1, gsz, gsz).float()
            zf = F.interpolate(zf, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            ys, xs = np.nonzero(weight > 0)
            y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, h)
            x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, w)
            people_px = weight[y0:y1, x0:x1] > 0
            panels = [img[y0:y1, x0:x1]]
            for d in (scene_depth, fused_depth, zf):
                sh = sd.shade(np.where(people_px, d[y0:y1, x0:x1], np.nan), f_px)
                panels.append(np.repeat(sh[..., None], 3, -1))
            Image.fromarray(np.concatenate(panels, 1).astype(np.uint8)).save(os.path.join(out, "people_shape.png"))
            log("people_shape.png: the picture | scene depth | with the people's own shape | the splats as made")
            if k is not None:
                kk = k.float().cpu().numpy()
                log(f"people's splats moved to their measured shape by {np.percentile(np.abs(kk - 1), 90) * 100:.1f} % "
                    "(90th percentile of the depth change)")
        except Exception as e:  # noqa: BLE001 - a diagnostic
            log(f"people shape picture skipped: {e}")
        return True

    # ---- frames
    step_f, step_b = (req.get("grid") or [1.5, 3])[:2]
    keep, steps = grid_index(float(step_f), float(step_b))
    keep_t = torch.from_numpy(keep).to(device)
    steps_t = torch.from_numpy(steps).to(device)[:, None]
    kf = int(req.get("keyframe_interval", 15))
    quant = meta.get("quant")
    camera = meta.get("camera")
    prev_q = prev_depth = prev_img = None
    corr = 1.0
    t0 = time.time()
    done_before = sum(1 for i in range(len(frames)) if os.path.exists(os.path.join(fdir, f"{i:06d}.bin")))
    first = True
    for i in range(len(frames)):
        path = os.path.join(fdir, f"{i:06d}.bin")
        if os.path.exists(path) and os.path.exists(path + ".json"):
            first = True               # resume: the next new frame must be a keyframe
            continue
        img = load(i)
        depth0, f_px, _ = geometry(img, fov)
        depth, ratio, weight = people(img, depth0, f_px)
        ok = np.isfinite(depth) & (depth > 0)
        depth = depth * scale
        # follow the previous frame where the picture did not change, drifting slowly back to the clip's scale
        r, still = stabilise(depth, prev_depth, img, prev_img, ok)
        corr = float(min(max((r if r is not None else corr) * 0.97 + 0.03, 0.8), 1.25))
        depth = depth * corr
        prev_depth, prev_img = depth, img
        g, info = sd.predict(predictor, img, f_px, device, torch, depth, ok)
        with torch.no_grad():
            from sharp.utils import color_space as cs

            all_means, all_scales = g.mean_vectors[0], g.singular_values[0]
            k = None
            if snap and weight is not None:          # the people exactly at their measured 3D shape
                all_means, all_scales, k = sd.snap_to_depth(g, depth, weight, torch)
            means = all_means[keep_t]
            scales = all_scales[keep_t] * steps_t
            quats = g.quaternions[0][keep_t]
            cols = cs.linearRGB2sRGB(g.colors[0][keep_t]).clamp(0, 1)
            op = g.opacities[0][keep_t]
            m, s_, q_, c_, o_ = (x.float().cpu().numpy() for x in (means, scales, quats, cols, op))
            if not sheet_done and weight is not None:
                sheet_done = people_sheet(img, depth0 * scale * corr, depth, all_means, g, weight, f_px, k)
        h, w = img.shape[:2]
        if quant is None:
            quant = codec.default_quant(fov, float(np.degrees(2 * np.arctan(h / 2 / f_px))))
            camera = {"width": w, "height": h, "fx": f_px, "fy": f_px, "cx": w / 2, "cy": h / 2}
            meta.update(quant=quant, camera=camera, count=int(len(keep)), grid=[float(step_f), float(step_b)])
            json.dump(meta, open(meta_path, "w"))
        arr = codec.quantize(m, s_, q_, c_, o_, quant)
        key = first or prev_q is None or i % kf == 0
        data = codec.encode_frame(arr, None if key else prev_q)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        json.dump({"key": bool(key), "stabilise": round(corr, 4), "still_share": round(still, 3),
                   "people": ratio is not None},
                  open(path + ".json", "w"))
        prev_q, first = arr, False
        n_done = i + 1
        el = time.time() - t0
        new = max(1, n_done - done_before)
        progress(0.05 + 0.95 * n_done / len(frames),
                 f"frame {n_done}/{len(frames)} · {el / new:.1f} s per frame · "
                 f"{(len(frames) - n_done) * el / new / 60:.0f} min left")
        if i % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        if i % 30 == 0:
            log(f"frame {n_done}: {len(data) / 2**20:.2f} MB, depth matched to the previous frame ×{corr:.3f} "
                f"({still:.0%} of the pixels unchanged)")
    emit("result", outputs={"frames_dir": fdir, "meta": meta_path}, frames=len(frames), count=int(len(keep)),
         camera=meta.get("camera"), quant=meta.get("quant"), scale=meta["scale"], scale_from=meta["scale_from"],
         fov_x=meta["fov_x"], people_shape=os.path.join(out, "people_shape.png") if sheet_done else None,
         vram_peak_mib=vram_peak_mib(torch))


if __name__ == "__main__":
    run(main)
