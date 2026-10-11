"""TEST-ONLY stand-in for workers/trellis_subject_worker.py: a small box as the 'subject' and a mask."""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
from _protocol import emit, progress, run  # noqa: E402


def main(req):
    import numpy as np
    from PIL import Image

    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    json.dump(req, open(os.path.join(out, "request.json"), "w"))
    v = np.array([[x, y, z] for x in (-.3, .3) for y in (-.5, .5) for z in (-.2, .2)], np.float32)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]], np.int32)
    np.savez(os.path.join(out, "subject.npz"), vertices=v, faces=f, colors=np.full_like(v, .5))
    with Image.open(req["image"]) as im:
        w, h = im.size
    m = np.zeros((h, w), np.uint8)
    m[h // 4: 3 * h // 4, w // 3: 2 * w // 3] = 255
    Image.fromarray(m).save(os.path.join(out, "subject_mask.png"))
    progress(1.0, "fake subject done")
    emit("result", outputs={"mesh": os.path.join(out, "subject.npz"), "mask": os.path.join(out, "subject_mask.png")},
         vertices=8, faces=4, pipeline_type=req.get("pipeline_type"), vram_peak_mib=None)


if __name__ == "__main__":
    run(main)
