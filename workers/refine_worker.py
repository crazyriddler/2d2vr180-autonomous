"""The photo's splat fitted to the photo itself (refine.py) - runtime 'render-cu124' (gsplat with its prebuilt
CUDA kernels; 2.0.0-rc8: in the photo engine gsplat could not run without a CUDA toolkit).

Request: {"ply": "locked.ply", "out": "locked_fitted.ply", "photo": "photo.png", "fx": 1245.2,
          "size": [W, H], "iters": 300}
The photo is read at any size (EXIF orientation applied) and compared with the splat seen from its camera
(principal point at the centre of ``size``, focal length ``fx`` in those pixels). Result: refine.refine_ply's
stats; "out" is written only when the fit improved.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402


def main(req):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    import refine

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="the fit to the photo needs CUDA (gsplat).")
        sys.exit(1)
    with Image.open(req["photo"]) as im:
        img = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    progress(0.05, "fitting the splat to the photo")
    st = refine.refine_ply(req["ply"], req["out"], img, float(req["fx"]), req["size"], torch, torch.device("cuda"),
                           iters=int(req.get("iters", 300)), log=log)
    progress(1.0, "done")
    emit("result", outputs={"fitted": req["out"] if st.get("used") else None}, stats=st,
         vram_peak_mib=vram_peak_mib(torch))


if __name__ == "__main__":
    run(main)
