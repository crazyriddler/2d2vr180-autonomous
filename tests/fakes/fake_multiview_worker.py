"""TEST-ONLY stand-in for workers/multiview_worker.py: writes a synthetic 3DGS
scene (with provenance) and the camera of every view in the real worker's format."""

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
    views = req["images"]
    if req.get("score_only"):
        # the first generated angle looks inconsistent; the others are fine
        sel = []
        for k, v in enumerate([v for v in views if v.get("generated")]):
            sc = 0.9 if k == 0 else 0.2
            sel.append({"view": v.get("label"), "chosen": os.path.basename(v["candidates"][0]),
                        "scores": {os.path.basename(c): sc for c in v["candidates"]}})
        with open(os.path.join(os.path.dirname(req["output_dir"]), "score_request.json"), "w") as f:
            json.dump(req, f)
        emit("result", score_only=True, camera_engine="vggt", candidate_selection=sel)
        return
    frames = [os.path.basename(v["path"]) for v in views]
    h, w, f = 90, 120, 100.0
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 3.0)
    pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    img = np.stack([xs * 2, ys * 2, np.full_like(xs, 128)], -1).astype(np.uint8)
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    sc = pointmap_to_gaussians(pts, img, None, Camera(w, h, f, f, w / 2, h / 2))
    gen = any(v.get("generated") for v in views)
    sc.provenance[:] = 0
    if gen:
        sc.provenance[: len(sc) // 3] = 2
    ply = write_gaussian_ply(sc, os.path.join(out, "scene.ply"), include_provenance=True)
    cams = [{"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
             "c2w": orbit_c2w(np.array([0, 0, 3.0]), 3.0, -10 + 20 * i / max(len(frames) - 1, 1), 0).tolist(),
             "file": fr, "generated": bool(views[i].get("generated"))} for i, fr in enumerate(frames)]
    with open(os.path.join(out, "cameras.json"), "w") as fh:
        json.dump(cams, fh)
    tt = os.path.join(out, "turntable")
    os.makedirs(tt, exist_ok=True)
    from PIL import Image
    for i in range(6):
        Image.fromarray(np.roll(img, i * 10, axis=1)).resize((128, 96)).save(os.path.join(tt, f"frame_{i:03d}.jpg"))
    progress(1.0, "fake multiview done")
    n = len(sc)
    emit("result", outputs={"ply": str(ply), "cameras": os.path.join(out, "cameras.json"), "turntable_frames": tt},
         vram_peak_mib=None,
         metric=True, views=len(views), real_views=sum(not v.get("generated") for v in views),
         provenance={"observed": n - (n // 3 if gen else 0), "inferred": 0, "generative": n // 3 if gen else 0})


if __name__ == "__main__":
    run(main)
