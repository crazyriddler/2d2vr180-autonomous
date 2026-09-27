"""Depth-Anything-V2-Small relative depth worker (runtime 'photo-cu128').

Request: {"model_dir": ".../depth-anything-v2-small", "images": [{"path", "out"}],
          "max_side": 1536, "allow_cpu": false}
Writes per image an .npz with disparity (H,W float32, relative inverse depth)
and image (H,W,3 uint8). Lifting to 3D happens in the application
(twod2vr180.rgbd.disparity_to_points) so the assumptions are reported there.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, progress, run, torch_env, vram_peak_mib  # noqa: E402


def main(req):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    env = torch_env(torch)
    if not env["cuda_available"] and not req.get("allow_cpu", False):
        emit("error", code="cuda_unavailable",
             message="CUDA is not available to PyTorch in this runtime (enable CPU mode in Settings to run slowly).")
        sys.exit(1)
    device = torch.device("cuda" if env["cuda_available"] else "cpu")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    progress(0.02, "loading Depth-Anything-V2")
    proc = AutoImageProcessor.from_pretrained(req["model_dir"], local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(req["model_dir"], local_files_only=True).to(device).eval()
    max_side = int(req.get("max_side", 1536))
    images = req["images"]
    for i, item in enumerate(images):
        with Image.open(item["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                s = max_side / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)
            inputs = proc(images=im, return_tensors="pt").to(device)
        with torch.inference_mode():
            pred = model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1).float(), size=arr.shape[:2], mode="bicubic",
                                                   align_corners=False)[0, 0]
        np.savez_compressed(item["out"], disparity=pred.cpu().numpy().astype(np.float32), image=arr)
        progress((i + 1) / len(images), f"depth {i + 1}/{len(images)}")
    emit("result", outputs=[it["out"] for it in images], vram_peak_mib=vram_peak_mib(torch),
         model="depth-anything-v2-small", metric=False)


if __name__ == "__main__":
    run(main)
