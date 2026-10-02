"""LaMa inpainting worker (runtime 'gen-cu128').

Request: {"model": ".../big-lama.pt", "items": [{"image": "a.png", "mask": "a_mask.png", "out": "a_out.png"}],
          "max_side": 2048}
Masks: white = fill. Only masked pixels are replaced; everything else is copied from the input.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, progress, run, torch_env, vram_peak_mib  # noqa: E402


def inpaint(model, img, mask, torch, device, max_side):
    """img (H,W,3) uint8, mask (H,W) bool → (H,W,3) uint8."""
    import numpy as np
    from PIL import Image

    H, W = mask.shape
    s = min(1.0, max_side / max(H, W))
    w, h = max(8, int(round(W * s / 8)) * 8), max(8, int(round(H * s / 8)) * 8)
    im = np.asarray(Image.fromarray(img).resize((w, h), Image.BICUBIC), np.float32) / 255
    mk = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    t_img = torch.from_numpy(im).permute(2, 0, 1)[None].to(device)
    t_mask = torch.from_numpy(mk.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        out = model(t_img, t_mask)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out * 255).round().astype(np.uint8)
    if (w, h) != (W, H):
        out = np.asarray(Image.fromarray(out).resize((W, H), Image.BICUBIC))
    res = img.copy()
    res[mask] = out[mask]
    return res


def main(req):
    import numpy as np
    import torch
    from PIL import Image

    env = torch_env(torch)
    device = torch.device("cuda" if env["cuda_available"] else "cpu")
    model = torch.jit.load(req["model"], map_location=device).eval()
    items = req["items"]
    for i, it in enumerate(items):
        img = np.asarray(Image.open(it["image"]).convert("RGB"))
        mask = np.asarray(Image.open(it["mask"]).convert("L")) > 127
        out = inpaint(model, img, mask, torch, device, int(req.get("max_side", 2048))) if mask.any() else img
        Image.fromarray(out).save(it["out"])
        progress((i + 1) / len(items), f"AI hole filling {i + 1}/{len(items)}")
    emit("result", outputs=[it["out"] for it in items], vram_peak_mib=vram_peak_mib(torch))


if __name__ == "__main__":
    run(main)
