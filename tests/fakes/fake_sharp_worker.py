"""TEST-ONLY stand-in for workers/sharp_worker.py: one splat per input image - its pixels on a plane 3 m
away, in the image's own camera frame - with the camera in the sidecar, as SHARP's PLY metadata gives it."""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
sys.path.insert(0, os.path.join(REPO, "src"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np
    from PIL import Image

    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
    os.makedirs(req["output_dir"], exist_ok=True)
    with open(os.path.join(os.path.dirname(req["output_dir"]), "fake_sharp_request.json"), "w") as f:
        json.dump(req, f)
    names = sorted(f for f in os.listdir(req["input_dir"]) if f.lower().endswith((".png", ".jpg")))
    outs = []
    for i, n in enumerate(names):
        with Image.open(os.path.join(req["input_dir"], n)) as im:
            focal = im.getexif().get_ifd(0x8769).get(0x920A)
            img = np.asarray(im.convert("RGB").resize((80, 60)))
        h, w = img.shape[:2]
        f = (float(focal) if focal else 30.0) * np.hypot(w, h) / np.hypot(36, 24)
        ys, xs = np.mgrid[0:h, 0:w]
        z = np.full((h, w), 3.0)
        pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
        cam = Camera(w, h, f, f, w / 2, h / 2)
        sc = pointmap_to_gaussians(pts, img, None, cam)
        sc.cameras = [cam]
        outs.append(str(write_gaussian_ply(sc, os.path.join(req["output_dir"], os.path.splitext(n)[0] + ".ply"))))
        progress((i + 1) / len(names), f"fake SHARP {i + 1}/{len(names)}")
    emit("result", outputs=outs, vram_peak_mib=None)


if __name__ == "__main__":
    run(main)
