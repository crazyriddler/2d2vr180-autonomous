"""TEST-ONLY stand-in for workers/render_worker.py: the app's CPU splat renderer, in the GPU worker's
output format (rgb, alpha, depth per view)."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
sys.path.insert(0, os.path.join(REPO, "src"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np

    from twod2vr.render import render_pinhole
    from twod2vr.scene import read_gaussian_ply

    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
    sc = read_gaussian_ply(req["ply"])
    os.makedirs(req["out_dir"], exist_ok=True)
    for i, v in enumerate(req["views"]):
        r = render_pinhole(sc, np.array(v["c2w"]), v["width"], v["height"], v["fx"], v["fy"], v["cx"], v["cy"],
                           crack_fill=False)
        np.savez(os.path.join(req["out_dir"], f"view_{i:06d}.npz"), rgb=r.rgb, alpha=r.covered.astype(np.float16),
                 depth=np.where(np.isfinite(r.depth), r.depth, 0).astype(np.float16))
        progress((i + 1) / len(req["views"]), "fake render")
    emit("result", outputs=len(req["views"]), vram_peak_mib=None)


if __name__ == "__main__":
    run(main)
