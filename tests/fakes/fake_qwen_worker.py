"""TEST-ONLY stand-in for workers/qwen_views_worker.py: "generates" each view by copying the photo."""

import json
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
    with open(os.path.join(out, f"stage_{req['stage']}.json"), "w") as f:
        json.dump(req, f)
    if req["stage"] == "encode":
        emit("result", stage="encode", views=len(req["views"]))
        return
    views = [{"path": req["image"], "generated": False}]
    for i, v in enumerate(req["views"]):
        paths = []
        for c in range(max(1, int(v.get("candidates", req.get("candidates", 1))))):
            p = os.path.join(out, f"view{i:02d}.png" if c == 0 else f"view{i:02d}_c{c}.png")
            shutil.copy(req["image"], p)
            paths.append(p)
        views.append({**v, "path": paths[0], "candidates": paths, "generated": True, "key": True})
    progress(1.0, "fake views done")
    emit("result", stage="generate", views=views, size=[160, 120], placement="stream", vram_peak_mib=None)


if __name__ == "__main__":
    run(main)
