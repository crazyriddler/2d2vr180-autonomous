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

    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
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
    emit("result", outputs=outputs, vram_peak_mib=None, f_px=float(K[0, 0]), size=[w, h], faces=[],
         face_scale=1.0, sharp_own_scale=1.2, depth_coverage=1.0)


if __name__ == "__main__":
    run(main)
