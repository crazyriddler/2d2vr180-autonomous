"""TEST-ONLY stand-in for workers/flashworld_worker.py: 'encode' writes an embedding file, 'generate'
writes a synthetic splat (a plane 2 m in front of the photo), cameras and turntable frames."""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
sys.path.insert(0, os.path.join(REPO, "src"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np

    emit("env", torch=None, cuda="12.4", cuda_available=True, device="fake")
    if req["stage"] == "encode":
        os.makedirs(os.path.dirname(req["embeds_path"]), exist_ok=True)
        with open(req["embeds_path"], "wb") as f:
            f.write(b"fake")
        with open(req["embeds_path"] + ".calls", "a") as f:      # how often the text encoder ran
            f.write("1\n")
        emit("result", embeds_path=req["embeds_path"])
        return
    os.makedirs(req["output_dir"], exist_ok=True)
    with open(os.path.join(req["output_dir"], "..", "fake_flashworld_generate.json"), "w") as f:
        json.dump(req, f)
    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    h, w, f = 64, 48, 60.0
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 2.0)
    pts = np.stack([(xs + .5 - w / 2) / f * z * 1.5, (ys + .5 - h / 2) / f * z * 1.5, z], -1)   # wider than the photo
    img = np.stack([xs * 4, ys * 3, np.full_like(xs, 128)], -1).astype(np.uint8)
    sc = pointmap_to_gaussians(pts, img, None, Camera(w, h, f, f, w / 2, h / 2))
    write_gaussian_ply(sc, os.path.join(out, "scene.ply"))
    with open(os.path.join(out, "cameras.json"), "w") as fh:
        json.dump([{"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
                    "c2w": np.eye(4).tolist(), "file": "photo.png", "generated": False}], fh)
    tt = os.path.join(out, "turntable")
    os.makedirs(tt, exist_ok=True)
    from PIL import Image
    for i in range(4):
        Image.fromarray(np.roll(img, i * 5, axis=1)).resize((64, 48)).save(os.path.join(tt, f"frame_{i:03d}.jpg"))
    progress(1.0, "fake flashworld done")
    emit("result", outputs={"ply": os.path.join(out, "scene.ply"), "cameras": os.path.join(out, "cameras.json"),
                            "turntable_frames": tt}, gaussians=len(sc), metric=True, subject_distance_m=2.0,
         hfov_deg=50.0, views=24, size=[480, 704])


if __name__ == "__main__":
    run(main)
