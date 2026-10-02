"""Generative novel views from one photo with Stable Virtual Camera (runtime 'gen-cu128').

The photo is the first view of a camera trajectory; the diffusion model
generates the other views (two-pass procedural sampling with a trajectory
prior, as in the upstream `img2trajvid_s-prob` task). The frames are then
reconstructed into a real 3D Gaussian splat by the multi-view engine.

Request:
  {"image": "...png", "output_dir": "...", "trajectory": "orbit" | "explore" | "spiral",
   "num_frames": 80, "steps": 50, "short_side": 576, "hfov_deg": 60 | null, "seed": 23,
   "seva_dir": dir with modelv1.1.safetensors, "vae_dir": dir with vae/, "clip_path": open_clip_model.safetensors}
Result: frames (input first) with OpenCV camera-to-world poses of the requested trajectory.
"""

import contextlib
import glob
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

TRAJECTORIES = ("arc", "orbit", "explore", "spiral")
PRESETS = {"arc": "lemniscate"}   # our name → Stable Virtual Camera preset (figure-of-eight, ±60°)


def short_sides(start: int, aspect: float, budget: int) -> list[int]:
    """Short sides (multiples of 64, as Stable Virtual Camera requires) to try, largest first,
    keeping width x height within ``budget`` pixels."""
    out = []
    for s in range(start - start % 64, 255, -64):
        if s * s * aspect <= budget * 1.02 and s not in out:
            out.append(s)
    return (out or [256])[:3]


def explore_c2ws(n, yaw_deg=70.0, pitch_deg=28.0, sway=0.1):
    """Look around from the capture position (figure-of-eight in yaw/pitch) with a small sway
    for parallax. Units: the subject is at depth ~1. Zero-mean translations and an identity
    first pose keep Stable Virtual Camera's camera normalisation at scale 1:1."""
    import numpy as np

    out = []
    for t in np.arange(n) / n:
        yaw = np.radians(yaw_deg) * np.sin(2 * np.pi * t)
        pitch = np.radians(pitch_deg) * np.sin(4 * np.pi * t)
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      # yaw about the camera's y (down) axis
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])       # pitch
        m = np.eye(4)
        m[:3, :3] = ry @ rx
        m[:3, 3] = [sway * np.sin(2 * np.pi * t), 0.3 * sway * np.sin(4 * np.pi * t), 0.0]
        out.append(m)
    return np.stack(out)


def intrinsics(hfov_deg, w, h, n):
    import numpy as np

    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return np.repeat(K[None], n, 0)


def trajectory(kind, n, hfov_deg, w, h):
    """(c2ws (n,4,4) OpenCV, Ks (n,3,3) pixels); view 0 is the input camera (identity)."""
    import numpy as np
    import torch

    if kind == "explore":
        return explore_c2ws(n), intrinsics(hfov_deg, w, h, n)
    from seva.geometry import get_preset_pose_fov

    c2ws, _ = get_preset_pose_fov(option=PRESETS.get(kind, kind), num_frames=n, start_w2c=torch.eye(4),
                                  look_at=torch.Tensor([0, 0, 10]))
    c2ws = np.asarray(c2ws, np.float64)
    if c2ws.shape[1] == 3:
        c2ws = np.concatenate([c2ws, np.tile([[[0, 0, 0, 1.0]]], (len(c2ws), 1, 1))], 1)
    return c2ws, intrinsics(hfov_deg, w, h, n)


class ProgressBar:
    """Stand-in for tqdm inside Stable Virtual Camera: tqdm redraws one console line with '\\r',
    which the app cannot see, so report chunks and diffusion steps as protocol events instead,
    with seconds per step and GPU memory (a full GPU spilling into shared system memory on
    Windows shows up as very slow steps)."""

    passes = 0          # chunk loops seen (first pass, second pass)
    chunk = (0, 1)      # (index, total) of the current chunk
    torch = None

    def __init__(self, iterable=None, total=None, desc="", **kw):
        self.it = iterable
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.desc = str(desc or "")
        self.n = 0
        self.t = time.time()
        self.sampling = "sampl" in self.desc.lower()   # chunk loops have no description
        if not self.sampling:
            ProgressBar.passes += 1
            ProgressBar.chunk = (0, self.total or 1)

    def __iter__(self):
        for x in self.it:
            yield x
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, k=1):
        self.n += k
        now = time.time()
        dt, self.t = now - self.t, now
        if not self.sampling:
            ProgressBar.chunk = (self.n, self.total or 1)
            return
        c, ct = ProgressBar.chunk
        lo, hi = (0.12, 0.3) if ProgressBar.passes <= 1 else (0.3, 0.97)
        frac = (c + self.n / max(self.total or 1, 1)) / max(ct, 1)
        mem = ""
        t = ProgressBar.torch
        if t is not None and t.cuda.is_available():
            used = t.cuda.memory_reserved() / 2**30
            tot = t.cuda.get_device_properties(0).total_memory / 2**30
            mem = f" · GPU memory {used:.1f}/{tot:.0f} GB"
            if dt > 20 and used > 0.95 * tot:
                log("GPU memory is full: Windows may be using shared system memory, which is very slow. "
                    "Close other GPU programs or use Fast mode.")
        progress(lo + (hi - lo) * min(frac, 1.0),
                 f"generating views: pass {min(ProgressBar.passes, 2)}/2, chunk {c + 1}/{ct}, "
                 f"step {self.n}/{self.total} · {dt:.1f} s/step{mem}")

    def close(self):
        pass

    def set_description(self, *a, **k):
        pass

    def set_postfix(self, *a, **k):
        pass

    def refresh(self, *a, **k):
        pass


def load_models(req, torch):
    import open_clip
    from diffusers.models import AutoencoderKL
    from torch import nn

    sys.modules.setdefault("gradio", types.SimpleNamespace(Progress=object))  # only used for type hints
    import seva.modules.transformer as tr

    # Upstream forces the flash-attention SDPA kernel, which Windows PyTorch builds lack;
    # let PyTorch choose (memory-efficient attention on Windows).
    tr.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model
    import seva.eval as seva_eval
    import seva.sampling as seva_sampling

    # Upstream low-VRAM mode: the diffusion model and the autoencoder take turns on the GPU
    # instead of sitting there together (peak memory = the larger one, not the sum).
    seva_eval.set_lowvram_mode(bool(req.get("lowvram", True)))

    ProgressBar.torch = torch
    seva_eval.tqdm = ProgressBar
    seva_sampling.tqdm = ProgressBar

    class LocalAE(AutoEncoder):
        def __init__(self, vae_dir):
            nn.Module.__init__(self)
            self.module = AutoencoderKL.from_pretrained(vae_dir, subfolder="vae", local_files_only=True)
            self.module.eval().requires_grad_(False)
            self.chunk_size = 1

    class LocalCLIP(CLIPConditioner):
        def __init__(self, path):
            nn.Module.__init__(self)
            self.module = open_clip.create_model_and_transforms("ViT-H-14", pretrained=path)[0]
            self.module.eval().requires_grad_(False)
            self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
            self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    model = SGMWrapper(load_model(model_version=1.1, pretrained_model_name_or_path=req["seva_dir"],
                                  weight_name="modelv1.1.safetensors", device="cpu", verbose=False).eval()).to("cuda")
    ae = LocalAE(req["vae_dir"]).to("cuda")
    clip = LocalCLIP(req["clip_path"]).float().eval()   # stays on the CPU: frees ~2.5 GB of VRAM

    class CPUConditioner(nn.Module):
        """CLIP image embedding computed on the CPU (only a handful of input images per chunk)."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            with torch.autocast("cuda", enabled=False), torch.no_grad():
                return self.inner(x.detach().float().cpu()).to(x.device)

        # Stable Virtual Camera moves every component to the GPU before use (seva.eval.load_model);
        # this one must stay on the CPU.
        def to(self, *a, **k):
            return self

        def cuda(self, *a, **k):
            return self

    return model, ae, CPUConditioner(clip), DiscreteDenoiser(num_idx=1000, device="cuda")


def generate(req, models, short_side, torch):
    import numpy as np
    from PIL import Image
    from seva.eval import infer_prior_stats, run_one_scene

    model, ae, clip, denoiser = models
    kind = req.get("trajectory", "orbit")
    n_targets = int(req.get("num_frames", 80))
    with Image.open(req["image"]) as im:
        W, H = im.size
    hfov = float(req.get("hfov_deg") or 60.0)
    c2ws, Ks = trajectory(kind, n_targets + 1, hfov, W, H)
    version_dict = {"H": 576, "W": 576, "T": 21, "C": 4, "f": 8, "options": {
        "chunk_strategy": "interp", "video_save_fps": 30.0, "beta_linear_start": 5e-6, "log_snr_shift": 2.4,
        "guider_types": [1, 2], "cfg": [4.0 if kind != "explore" else 3.0, 2.0], "camera_scale": 2.0,
        "num_steps": int(req.get("steps", 50)), "cfg_min": 1.2, "encoding_t": 1, "decoding_t": 1,
        "replace_or_include_input": True, "L_short": short_side, "num_targets": n_targets,
        "use_traj_prior": True, "traj_prior": PRESETS.get(kind, kind), "save_input": True}}
    num_anchors = infer_prior_stats(version_dict["T"], 1, num_total_frames=n_targets, version_dict=version_dict)
    anchor_idx = [round(i) for i in np.linspace(1, n_targets, num_anchors)]
    save = os.path.join(req["output_dir"], f"seva_{short_side}")
    t_c2w = torch.tensor(c2ws[:, :3], dtype=torch.float32)
    t_K = torch.tensor(Ks, dtype=torch.float32)
    gen = run_one_scene(
        "img2trajvid_s-prob", version_dict, model=model, ae=ae, conditioner=clip, denoiser=denoiser,
        image_cond={"img": [req["image"]] + [None] * n_targets, "input_indices": [0],
                    "prior_indices": [float(i) for i in np.linspace(1, n_targets, num_anchors)]},
        camera_cond={"c2w": t_c2w.clone(), "K": t_K.clone(), "input_indices": list(range(n_targets + 1))},
        save_path=save, use_traj_prior=True, traj_prior_Ks=t_K[anchor_idx].clone(),
        traj_prior_c2ws=t_c2w[anchor_idx].clone(), seed=int(req.get("seed", 23)))
    ProgressBar.passes = 0
    for _ in gen:
        pass
    frames = sorted(glob.glob(os.path.join(save, "samples-rgb", "*.png")))
    inputs = sorted(glob.glob(os.path.join(save, "input", "*.png")))
    if len(frames) == n_targets and inputs:
        frames = inputs[:1] + frames
    if len(frames) != n_targets + 1:
        raise RuntimeError(f"expected {n_targets + 1} frames, got {len(frames)}")
    w, h = Image.open(frames[0]).size
    return frames, c2ws, intrinsics(hfov, w, h, len(frames)), (w, h)


def main(req):
    import json

    import torch

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="Generating views needs an NVIDIA GPU (CUDA).")
        sys.exit(1)
    if req.get("trajectory", "orbit") not in TRAJECTORIES:
        emit("error", code="bad_input", message=f"unknown trajectory {req.get('trajectory')}")
        sys.exit(1)
    os.makedirs(req["output_dir"], exist_ok=True)
    progress(0.02, "loading Stable Virtual Camera")
    models = load_models(req, torch)
    progress(0.1, "models loaded")
    # Never let Windows spill GPU memory into shared system RAM (10-50x slower): cap the
    # allocator below the card's size so an out-of-memory error triggers a smaller retry instead.
    torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.92)))
    from PIL import Image

    with Image.open(req["image"]) as im:
        aspect = max(im.size) / min(im.size)
    # Start at the model's native 576 px (best quality); the memory cap turns a too-large attempt
    # into an out-of-memory error and the next, smaller size is tried.
    budget = int(req.get("pixel_budget", 0)) or int(576 * 576 * max(aspect, 1.0))
    sizes = short_sides(int(req.get("short_side", 576)), aspect, budget) + [384, 320]
    sizes = list(dict.fromkeys(sizes))
    log(f"generation resolutions to try (short side): {sizes}")
    last = None
    for s in sizes:
        try:
            frames, c2ws, Ks, wh = generate(req, models, s, torch)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # retry smaller
            if "out of memory" not in str(e).lower():
                raise
            last = e
            import gc

            gc.collect()
            log(f"out of GPU memory at {s} px; retrying at a lower resolution")
            torch.cuda.empty_cache()
    else:
        raise last
    out = [{"path": p, "generated": i != 0, "c2w": c2ws[i].tolist(), "K": Ks[i].tolist()}
           for i, p in enumerate(frames)]
    with open(os.path.join(req["output_dir"], "views.json"), "w") as f:
        json.dump(out, f)
    emit("result", views=out, size=list(wh), vram_peak_mib=vram_peak_mib(torch), trajectory=req.get("trajectory"))


if __name__ == "__main__":
    run(main)
