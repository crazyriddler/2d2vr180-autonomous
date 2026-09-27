"""GPU Gaussian-splat renderer (gsplat) — runtime 'photo-cu128'.

Request: {"ply": "scene.ply", "views": [{"c2w": 4x4, "width", "height", "fx", "fy", "cx", "cy"}],
          "out_dir": "...", "batch": 10}
Writes out_dir/view_{i:06d}.npz with rgb (H,W,3 uint8, premultiplied by alpha),
alpha (H,W float16) and depth (H,W float16, expected z). True 3DGS alpha
compositing with spherical harmonics when the PLY carries them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, progress, run, torch_env, vram_peak_mib  # noqa: E402


def load_ply(path, torch, device):
    import numpy as np
    from plyfile import PlyData

    v = PlyData.read(path)["vertex"].data
    names = v.dtype.names
    f = lambda *ks: np.stack([np.asarray(v[k], np.float32) for k in ks], 1)  # noqa: E731
    means = f("x", "y", "z")
    scales = np.exp(f("scale_0", "scale_1", "scale_2"))
    quats = f("rot_0", "rot_1", "rot_2", "rot_3")
    opac = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32)))
    dc = f("f_dc_0", "f_dc_1", "f_dc_2")
    rest = sorted([k for k in names if k.startswith("f_rest_")], key=lambda s: int(s[7:]))
    if rest:
        r = f(*rest).reshape(len(means), 3, -1).transpose(0, 2, 1)  # INRIA: channel-major
        sh = np.concatenate([dc[:, None, :], r], 1)
        degree = int(round(np.sqrt(sh.shape[1]))) - 1
    else:
        sh = dc[:, None, :]
        degree = 0
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return t(means), t(quats), t(scales), t(opac), t(sh), degree


def main(req):
    import numpy as np
    import torch

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="gsplat rendering needs CUDA.")
        sys.exit(1)
    from gsplat import rasterization

    dev = torch.device("cuda")
    means, quats, scales, opac, sh, degree = load_ply(req["ply"], torch, dev)
    views = req["views"]
    os.makedirs(req["out_dir"], exist_ok=True)
    batch = int(req.get("batch", 10))
    groups = {}
    for i, vw in enumerate(views):
        groups.setdefault((vw["width"], vw["height"]), []).append(i)
    done = 0
    for (w, h), idx in groups.items():
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            c2w = torch.tensor(np.array([views[i]["c2w"] for i in chunk]), dtype=torch.float32, device=dev)
            viewmats = torch.linalg.inv(c2w)
            Ks = torch.tensor([[[views[i]["fx"], 0, views[i]["cx"]], [0, views[i]["fy"], views[i]["cy"]], [0, 0, 1]]
                               for i in chunk], dtype=torch.float32, device=dev)
            with torch.inference_mode():
                out, alpha, _ = rasterization(means, quats, scales, opac, sh, viewmats, Ks, w, h,
                                              sh_degree=degree, near_plane=0.01, render_mode="RGB+ED")
            rgb = (out[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            dep = out[..., 3].float().cpu().numpy()
            al = alpha[..., 0].float().cpu().numpy()
            for k, i in enumerate(chunk):
                np.savez(os.path.join(req["out_dir"], f"view_{i:06d}.npz"), rgb=rgb[k],
                         alpha=al[k].astype(np.float16), depth=dep[k].astype(np.float16))
            done += len(chunk)
            progress(done / len(views), f"GPU render {done}/{len(views)}")
    emit("result", outputs=len(views), vram_peak_mib=vram_peak_mib(torch))


if __name__ == "__main__":
    run(main)
