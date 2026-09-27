"""TEST-ONLY stand-in for workers/moge_worker.py.

Speaks the real worker protocol but fabricates a smooth synthetic depth map
instead of running MoGe-2, so the whole job pipeline (worker process, events,
cancellation, export, VR180, report) can be tested on machines without a GPU
or model weights. Never shipped; never registered as a backend.
"""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np
    from PIL import Image

    emit("env", torch=None, cuda=None, cuda_available=False, device="fake-test-worker")
    delay = float(os.environ.get("FAKE_WORKER_DELAY", "0"))
    if os.environ.get("FAKE_WORKER_FAIL"):
        emit("error", code=os.environ["FAKE_WORKER_FAIL"], message="simulated failure")
        sys.exit(1)
    for i, item in enumerate(req["images"]):
        time.sleep(delay)
        arr = np.asarray(Image.open(item["path"]).convert("RGB"))
        h, w = arr.shape[:2]
        lum = arr.mean(-1) / 255.0
        ys, xs = np.mgrid[0:h, 0:w]
        z = 2.0 + 1.0 * (1.0 - lum)
        f = 0.9 * w
        pts = np.stack([(xs + 0.5 - w / 2) / f * z, (ys + 0.5 - h / 2) / f * z, z], -1).astype(np.float32)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        if "model_dir" in req:  # Depth-Anything protocol: relative disparity only
            np.savez_compressed(item["out"], disparity=(1.0 / z).astype(np.float32), image=arr)
        else:
            np.savez_compressed(item["out"], points=pts, mask=np.ones((h, w), bool), intrinsics=K, image=arr)
        progress((i + 1) / len(req["images"]), f"fake geometry {i + 1}")
    emit("result", outputs=[it["out"] for it in req["images"]], vram_peak_mib=None, model="fake", metric=True)


if __name__ == "__main__":
    run(main)
