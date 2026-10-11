"""MoGe-2 / MoGe-3 monocular metric geometry worker (runs in runtime 'photo-cu128').

Request:
  {"model_path": ".../model.pt", "images": [{"path": "...", "out": ".../frame.npz", "fov_x": 67.4}],
   "max_side": 1536, "fp16": true, "resolution_level": 9, "version": "v2" | "v3", "refine_steps": 3}
``fov_x`` (optional, degrees): the photo's real horizontal field of view (EXIF), which fixes the depth/size
ratio of the whole result. MoGe-3 (July 2026) refines MoGe-2's estimate in 3D (sparse volumetric
refinement through FlexGEMM, run by Triton on Windows) for finer, more accurate geometry.
Writes per image an .npz with points (H,W,3 float32, OpenCV camera frame, metres),
mask (H,W bool), intrinsics (3x3, pixels), image (H,W,3 uint8).
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
             message="CUDA is not available to PyTorch in this runtime (driver too old or no NVIDIA GPU).")
        sys.exit(1)
    device = torch.device("cuda" if env["cuda_available"] else "cpu")
    version = req.get("version", "v2")
    if version == "v3":
        os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "never")
        import flex_gemm

        flex_gemm.config.AUTOTUNE_MODE = "never"     # as MoGe's own app: no slow first-call autotuning
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel

    progress(0.02, f"loading MoGe-{version[1:]}")
    model = MoGeModel.from_pretrained(req["model_path"]).to(device).eval()
    images = req["images"]
    max_side = int(req.get("max_side", 1536))
    for i, item in enumerate(images):
        with Image.open(item["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                s = max_side / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)
        t = torch.from_numpy(arr).to(device).float().div(255).permute(2, 0, 1)
        kw = {"refine_steps": int(req.get("refine_steps", 3))} if version == "v3" else {}
        out = model.infer(t, resolution_level=int(req.get("resolution_level", 9)), fov_x=item.get("fov_x"),
                          use_fp16=bool(req.get("fp16", True)) and device.type == "cuda", **kw)
        h, w = arr.shape[:2]
        K = out["intrinsics"].float().cpu().numpy().copy()
        K[0] *= w  # MoGe returns normalised intrinsics
        K[1] *= h
        np.savez_compressed(
            item["out"],
            points=out["points"].float().cpu().numpy().astype(np.float32),
            mask=out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.ones((h, w), bool),
            intrinsics=K.astype(np.float64),
            image=arr,
        )
        progress((i + 1) / len(images), f"geometry {i + 1}/{len(images)}")
        if device.type == "cuda" and (i % 8 == 7):
            torch.cuda.empty_cache()
    emit("result", outputs=[it["out"] for it in images], vram_peak_mib=vram_peak_mib(torch),
         model=f"moge-{version[1:]}", metric=True)


if __name__ == "__main__":
    run(main)
