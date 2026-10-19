"""TEST-ONLY stand-in for workers/sharp_depth_worker.py: the photo's (fake) MoGe points as splats - "locked"
at the given depth, "plain" 20 % deeper (SHARP's own guess) - in SHARP's file layout with the camera."""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
sys.path.insert(0, os.path.join(REPO, "src"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np

    from twod2vr.rgbd import pointmap_to_gaussians
    from twod2vr.scene import Camera, write_gaussian_ply

    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
    if req.get("mode") == "fill":      # each filled view: its known depth, the holes 10 % behind the farthest
        from PIL import Image

        stats = []
        for v in req["views"]:
            img = np.asarray(Image.open(v["image"]).convert("RGB"))
            hole = np.asarray(Image.open(v["mask"]).convert("L")) > 127
            with np.load(v["known"]) as z:
                d = z["depth"].astype(np.float64)
            known = np.isfinite(d) & (d > 0)
            d = np.where(known, d, np.max(d[known]) * 1.1 if known.any() else 3.0)
            h, w = d.shape
            cam = Camera(w, h, v["fx"], v["fy"], v["cx"], v["cy"])
            ys, xs = np.mgrid[0:h, 0:w]
            pts = np.stack([(xs + .5 - cam.cx) / cam.fx * d, (ys + .5 - cam.cy) / cam.fy * d, d], -1)
            sc = pointmap_to_gaussians(pts, img, None, cam)
            sc.cameras = [cam]
            write_gaussian_ply(sc, v["out"])
            stats.append({"hole_fraction": float(hole.mean()), "ring_pixels": 0, "scale": 1.0, "shift": 0.0})
        with open(os.path.join(os.path.dirname(req["views"][0]["out"]), "fake_fill_request.json"), "w") as f:
            json.dump(req, f)
        emit("result", outputs=[v["out"] for v in req["views"]], stats=stats, vram_peak_mib=None)
        return
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fake_sharp_depth_request.json"), "w") as f:
        json.dump(req, f)
    g = np.load(req["geometry"])
    pts, img, K = g["points"], g["image"], g["intrinsics"]
    h, w = img.shape[:2]
    cam = Camera(w, h, float(K[0, 0]), float(K[1, 1]), w / 2, h / 2)
    outputs = {}
    for name, s in (("locked", 1.0), ("plain", 1.2)):
        sc = pointmap_to_gaussians(pts * s, img, None, cam)
        sc.cameras = [cam]
        outputs[name] = str(write_gaussian_ply(sc, os.path.join(out, f"{name}.ply")))
        if not req.get("plain", True):
            break
    progress(1.0, "fake SHARP with depth")
    people = None
    if req.get("human_seg") and req.get("human_pointmap"):
        people = {"people_fraction": 0.3, "box": [0, h, 0, w], "scene_vs_human_distance": 1.1,
                  "human_model": "sapiens2_1b"}
    detail = None
    if req.get("detail"):        # maximum detail: the same splats twice as dense (a stand-in), the normal kept
        import shutil

        outputs["standard"] = outputs["locked"]
        sc = pointmap_to_gaussians(pts, img, None, cam)
        jit = np.random.default_rng(0).normal(0, 0.002, sc.means.shape).astype(np.float32)
        both = sc.subset(np.r_[np.arange(len(sc)), np.arange(len(sc))])
        both.means = both.means + np.r_[np.zeros_like(jit), jit]
        both.cameras = [cam]
        outputs["locked"] = str(write_gaussian_ply(both, os.path.join(out, "locked_detail.ply")))
        detail = {"work_size": [w, h], "tiles": 1, "splats": len(both), "placed_share": 0.9}
        shutil.copy(req["photo"], os.path.join(out, "detail_photo_seen.png"))
    precision = None
    if req.get("precision"):      # rc7: the relief sheet and what each precision step did
        from PIL import Image

        Image.fromarray(img).save(os.path.join(out, "relief.png"))
        outputs["relief"] = os.path.join(out, "relief.png")
        precision = {"normals": {"used": True, "rms_change": 0.004, "cutoff_px": 15.4},
                     "edges": {"band_share": 0.03}, "placed_share": 0.95,
                     "fit": {"used": True, "psnr_before": 27.0, "psnr_after": 31.5} if req.get("refine") else None,
                     "people_normals": bool(req.get("human_normal"))}
    emit("result", outputs=outputs, vram_peak_mib=None, f_px=float(K[0, 0]), size=[w, h], faces=[], detail=detail,
         precision=precision,
         face_scale=1.0, sharp_own_scale=1.2, depth_coverage=1.0, people=people,
         scale=1 / 1.1 if people else 1.0, scale_from="human model" if people else "measured depth",
         ipd_target_m=req.get("ipd_target_m", 0.063))


if __name__ == "__main__":
    run(main)
