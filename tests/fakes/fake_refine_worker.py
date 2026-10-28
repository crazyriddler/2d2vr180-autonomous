"""TEST-ONLY stand-in for workers/refine_worker.py: copies the splat and reports a better fit."""

import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    emit("env", torch=None, cuda="12.4", cuda_available=True, device="fake")
    with open(os.path.join(os.path.dirname(req["out"]), "fake_refine_request.json"), "w") as f:
        json.dump(req, f)
    shutil.copy(req["ply"], req["out"])
    progress(1.0, "fake fit")
    emit("result", outputs={"fitted": req["out"]}, vram_peak_mib=None,
         stats={"used": True, "iterations": req.get("iters", 300), "psnr_before": 27.0, "psnr_after": 31.5})


if __name__ == "__main__":
    run(main)
