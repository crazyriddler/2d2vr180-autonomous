"""TEST-ONLY stand-in for workers/seva_worker.py: "generates" views by copying the
input photo, in the real worker's result format."""

import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    views = []
    for i in range(int(req.get("num_frames") or req.get("frames")) + 1):
        p = os.path.join(out, f"view_{i:03d}.png")
        shutil.copy(req["image"], p)
        views.append({"path": p, "generated": i != 0, "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                      "K": [[100, 0, 80], [0, 100, 60], [0, 0, 1]]})
    progress(1.0, "fake generation done")
    emit("result", views=views, size=[160, 120], vram_peak_mib=None, trajectory=req["trajectory"])


if __name__ == "__main__":
    run(main)
