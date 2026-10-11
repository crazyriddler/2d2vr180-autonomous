"""TEST-ONLY stand-in for workers/video4d_worker.py: a 4D scene in the real worker's file formats from
the frames it is given. The background is the per-pixel median of the frames on a plane 3 m away;
pixels that differ from it become per-frame splats 2 m away (what moves); the camera is fixed, or slides
sideways when the request says the camera moves."""

import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
sys.path.insert(0, os.path.join(REPO, "src"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np
    from PIL import Image

    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    emit("env", torch=None, cuda="12.4", cuda_available=True, device="fake")
    if os.environ.get("FAKE_WORKER_FAIL"):
        emit("error", code=os.environ["FAKE_WORKER_FAIL"], message="simulated failure")
        sys.exit(1)
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in req["frames"]]
    n = len(frames)
    h, w = frames[0].shape[:2]
    f = 0.9 * w
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    delay = float(os.environ.get("FAKE_WORKER_DELAY", "0"))
    for i in range(n):
        time.sleep(delay / max(n, 1))
        progress(0.8 * (i + 1) / n, f"fake 4D frame {i + 1}/{n}")
    stack = np.stack(frames).astype(np.int16)
    median = np.median(stack, 0).astype(np.uint8)
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 3.0)
    pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    bg = pointmap_to_gaussians(pts, median, None, Camera(w, h, f, f, w / 2, h / 2))
    bg.provenance[:] = 1
    bg_path = os.path.join(out, "background.ply")
    write_gaussian_ply(bg, bg_path, include_provenance=True)
    moving = req.get("camera") == "moving"
    keys = {k: [] for k in ("means", "scales", "quats", "opacity", "colors", "t_center", "t_scale")}
    fractions = []
    for i, fr in enumerate(frames):
        m = np.abs(stack[i] - median).max(-1) > 60
        fractions.append(float(m.mean()))
        yy, xx = np.nonzero(m)
        k = len(yy)
        if not k:
            continue
        zz = np.full(k, 2.0)
        keys["means"].append(np.stack([(xx + .5 - w / 2) / f * zz, (yy + .5 - h / 2) / f * zz, zz], 1))
        keys["scales"].append(np.full((k, 3), 0.75 * 2.0 / f))
        keys["quats"].append(np.tile([1.0, 0, 0, 0], (k, 1)))
        keys["opacity"].append(np.full(k, 0.95))
        keys["colors"].append(fr[yy, xx] / 255.0)
        keys["t_center"].append(np.full(k, (i + 0.5) / n))
        keys["t_scale"].append(np.full(k, 0.45 / n))
    shapes = {"means": (0, 3), "scales": (0, 3), "quats": (0, 4), "colors": (0, 3)}
    dyn = {k: (np.concatenate(v) if v else np.zeros(shapes.get(k, (0,)))).astype(np.float32) for k, v in keys.items()}
    smooth = bool(req.get("motion")) and os.environ.get("FAKE_4D_MOTION") and len(keys["means"]) > 1
    if smooth:     # "smooth motion": the first frame's movers travel to where the last frame's are
        first, last = keys["means"][0], keys["means"][-1]
        k = len(first)
        dyn = {"means": first.astype(np.float32), "scales": keys["scales"][0].astype(np.float32),
               "quats": keys["quats"][0].astype(np.float32), "opacity": keys["opacity"][0].astype(np.float32),
               "colors": keys["colors"][0].astype(np.float32), "t_center": np.full(k, 0.5 / n, np.float32),
               "t_scale": np.full(k, 10.0, np.float32), "omega": np.zeros((k, 4), np.float32)}
        motion = np.zeros((k, 9), np.float32)
        motion[:, :3] = (last.mean(0) - first.mean(0)) / ((n - 1) / n)
        dyn["motion"] = motion
    dyn["provenance"] = np.ones(len(dyn["means"]), np.uint8)
    np.savez(os.path.join(out, "dynamic.npz"), **dyn)
    cams = []
    for i in range(n):
        c2w = np.eye(4)
        if moving:
            c2w[0, 3] = 0.02 * i
        cams.append({"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "c2w": c2w.tolist(),
                     "file": os.path.basename(req["frames"][i]), "generated": False})
    with open(os.path.join(out, "cameras.json"), "w") as fh:
        json.dump(cams, fh)
    Image.fromarray(median).save(os.path.join(out, "moving_preview.png"))
    progress(1.0, "fake 4D done")
    emit("result", outputs={"background": bg_path, "dynamic": os.path.join(out, "dynamic.npz"),
                            "cameras": os.path.join(out, "cameras.json"),
                            "moving_preview": os.path.join(out, "moving_preview.png")},
         vram_peak_mib=None, metric=True, metric_scale_factor=1.0, frames=n, fps=req["fps"],
         camera_motion="moving" if moving else "fixed", moving_fraction=[round(x, 4) for x in fractions],
         moving_model="motion" if smooth else "per_frame",
         background_splats=len(bg), moving_splats=int(len(dyn["means"])), reference_psnr_db=30.0, mesh=None,
         provenance={"observed": 0, "inferred": len(bg) + int(len(dyn["means"]))})


if __name__ == "__main__":
    run(main)
