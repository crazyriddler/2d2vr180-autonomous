"""TEST-ONLY stand-in for workers/recon3d_worker.py: writes a synthetic 3DGS
scene and a short camera path in the real worker's output format."""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
sys.path.insert(0, os.path.join(REPO, "src"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np

    from twod2vr180.render import orbit_c2w
    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    emit("env", torch=None, cuda="12.4", cuda_available=True, device="fake")
    frames = sorted(os.listdir(req["image_dir"]))
    h, w, f = 90, 120, 100.0
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 3.0)
    pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    img = np.stack([xs * 2, ys * 2, np.full_like(xs, 128)], -1).astype(np.uint8)
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    ply = write_gaussian_ply(pointmap_to_gaussians(pts, img, None, Camera(w, h, f, f, w / 2, h / 2)),
                             os.path.join(out, "scene.ply"))
    cams = [{"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
             "c2w": orbit_c2w(np.array([0, 0, 3.0]), 3.0, -10 + 20 * i / max(len(frames) - 1, 1), 0).tolist(),
             "file": fr} for i, fr in enumerate(frames)]
    with open(os.path.join(out, "cameras.json"), "w") as fh:
        json.dump(cams, fh)
    progress(1.0, "fake recon3d done")
    emit("result", outputs={"ply": str(ply), "cameras": os.path.join(out, "cameras.json")}, vram_peak_mib=None,
         metric=True)


if __name__ == "__main__":
    run(main)
