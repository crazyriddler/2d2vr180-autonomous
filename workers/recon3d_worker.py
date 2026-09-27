"""recon3d video reconstruction worker (runs in runtime 'recon3d-cu124').

Request: {"image_dir": "...", "output_dir": "...", "max_frames": 80, "steps": 7000,
          "resize": 960, "chunk_size": 20, "mesh": true, "metric": true}
Result: scene.ply (+ .splat, .obj) and cameras.json (OpenCV c2w, first frame = world).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402


def main(req):
    import numpy as np
    import torch

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="recon3d needs CUDA (gsplat training).")
        sys.exit(1)
    from recon3d.gaussian_train import TrainConfig
    from recon3d.pipeline import PipelineConfig, reconstruct

    vram_gib = env.get("vram_total_mib", 0) / 1024
    chunk = int(req.get("chunk_size") or (20 if vram_gib >= 15 else 12))
    cfg = PipelineConfig(max_frames=int(req.get("max_frames", 80)), target_fps=None,
                         resize_long_edge=int(req.get("resize", 960)), chunk_size=chunk,
                         metric_align=bool(req.get("metric", True)), export_mesh=bool(req.get("mesh", True)),
                         launch_viewer=False,
                         train_config=TrainConfig(max_steps=int(req.get("steps", 7000))))
    progress(0.05, f"recon3d: {req['image_dir']} (chunk {chunk})")
    out = req["output_dir"]
    ply = reconstruct(req["image_dir"], out, cfg)
    cams = []
    tf = os.path.join(out, "transforms.json")
    if os.path.exists(tf):
        with open(tf) as f:
            t = json.load(f)
        for fr in t["frames"]:
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV (undo recon3d's export flip)
            cams.append({"width": t["w"], "height": t["h"], "fx": t["fl_x"], "fy": t["fl_y"],
                         "cx": t["cx"], "cy": t["cy"], "c2w": c2w.tolist(), "file": fr["file_path"]})
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(cams, f)
    outputs = {"ply": str(ply), "cameras": os.path.join(out, "cameras.json")}
    for k, name in (("splat", "scene.splat"), ("obj", "scene.obj")):
        p = os.path.join(out, name)
        if os.path.exists(p):
            outputs[k] = p
    emit("result", outputs=outputs, vram_peak_mib=vram_peak_mib(torch), metric=bool(req.get("metric", True)))


if __name__ == "__main__":
    run(main)
