"""TEST-ONLY stand-in for workers/inpaint_worker.py: paints masked pixels magenta."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
from _protocol import emit, run  # noqa: E402


def main(req):
    import numpy as np
    from PIL import Image

    for it in req["items"]:
        img = np.asarray(Image.open(it["image"]).convert("RGB")).copy()
        img[np.asarray(Image.open(it["mask"])) > 127] = [255, 0, 255]
        Image.fromarray(img).save(it["out"])
    emit("result", outputs=[it["out"] for it in req["items"]])


if __name__ == "__main__":
    run(main)
