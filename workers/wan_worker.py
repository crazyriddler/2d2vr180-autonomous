"""Generative camera moves from one photo with Wan 2.2 Fun 5B Control-Camera (runtime 'gen-cu128').

A video diffusion model (Alibaba PAI, Apache-2.0) trained on large amounts of real footage -
much better than Stable Virtual Camera at keeping people and animals intact. Each "shot" starts
at the user's photo and follows a camera path given as per-frame poses (CameraCtrl format);
several shots (e.g. one to the left, one to the right) are generated and their frames are then
reconstructed into one 3D splat by the multi-view engine.

Request:
  {"image": "...", "output_dir": "...", "model_dir": "...", "trajectory": "arc" | "orbit" | "explore" | "spiral",
   "frames": 49, "steps": 50, "short_side": 704, "seed": 42, "prompt": "", "every": 2}
"""

import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

TRAJECTORIES = ("arc", "orbit", "explore", "spiral")

PROMPT = ("A still moment captured on camera; the camera slowly moves around the scene while everything in it "
          "stays perfectly still, frozen in time. Realistic, sharp, highly detailed, natural anatomy, the same "
          "person and the same clothes throughout, consistent lighting.")
NEGATIVE = ("moving people, walking, talking, gesturing, changing pose, changing expression, extra limbs, extra "
            "arms, extra legs, extra fingers, deformed face, distorted body, mutation, blurry, motion blur, "
            "flicker, text, subtitles, watermark, comic, painting, cartoon, low quality, worst quality, "
            "JPEG artifacts, static camera")

# Wan 2.2 Fun 5B (config/wan2.2/wan_civitai_5b.yaml upstream)
CONFIG = {
    "format": "civitai", "pipeline": "Wan",
    "transformer_additional_kwargs": {"transformer_low_noise_model_subpath": "./",
                                      "transformer_combination_type": "single",
                                      "dict_mapping": {"in_dim": "in_channels", "dim": "hidden_size"}},
    "vae_kwargs": {"vae_type": "AutoencoderKLWan3_8", "vae_subpath": "Wan2.2_VAE.pth",
                   "temporal_compression_ratio": 4, "spatial_compression_ratio": 16},
    "text_encoder_kwargs": {"text_encoder_subpath": "models_t5_umt5-xxl-enc-bf16.pth",
                            "tokenizer_subpath": "google/umt5-xxl", "text_length": 512, "vocab": 256384,
                            "dim": 4096, "dim_attn": 4096, "dim_ffn": 10240, "num_heads": 64, "num_layers": 24,
                            "num_buckets": 32, "shared_pos": False, "dropout": 0.0},
    "scheduler_kwargs": {"scheduler_subpath": None, "num_train_timesteps": 1000, "shift": 5.0,
                         "use_dynamic_shifting": False, "base_shift": 0.5, "max_shift": 1.15,
                         "base_image_seq_len": 256, "max_image_seq_len": 4096},
}


# ----------------------------------------------------------------------------- camera paths
def _yaw_pitch(yaw, pitch):
    import numpy as np

    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


def orbit_c2w(theta, radius=2.0, pitch=0.0):
    """Camera on a circle around the point (0, 0, radius) in front of the first camera (OpenCV)."""
    import numpy as np

    m = np.eye(4)
    m[:3, :3] = _yaw_pitch(-theta, pitch)
    m[:3, 3] = [radius * math.sin(theta), 0.0, radius * (1 - math.cos(theta))]
    return m


def shots(kind, n):
    """List of shots; each is a list of n camera-to-world matrices starting at the photo (identity)."""
    import numpy as np

    t = np.linspace(0, 1, n)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)            # smooth start and stop
    if kind == "arc":
        return [[orbit_c2w(s * math.radians(45) * e) for e in ease] for s in (1, -1)]
    if kind == "orbit":
        return [[orbit_c2w(s * math.radians(100) * e) for e in ease] for s in (1, -1)]
    if kind == "explore":
        out = []
        for yaw, pitch in ((60, 0), (-60, 0), (0, 30), (0, -25)):
            shot = []
            for e in ease:
                m = np.eye(4)
                m[:3, :3] = _yaw_pitch(math.radians(yaw) * e, math.radians(pitch) * e)
                m[:3, 3] = [0.15 * math.sin(math.radians(yaw)) * e, 0.0, 0.1 * e]
                shot.append(m)
            out.append(shot)
        return out
    # spiral: small forward-facing circle with a slight push-in
    shot = []
    for k, e in enumerate(t):
        m = np.eye(4)
        a = 2 * math.pi * e
        m[:3, 3] = [0.25 * math.sin(a) * e, 0.15 * (1 - math.cos(a)) * e, 0.3 * e]
        m[:3, :3] = _yaw_pitch(-0.12 * math.sin(a) * e, 0.08 * (1 - math.cos(a)) * e)
        shot.append(m)
    return [shot]


def pose_rows(c2ws, hfov_deg, w, h):
    """CameraCtrl rows: [frame, fx, fy, cx, cy, 0, 0, w2c (3x4 row-major)] with normalised intrinsics."""
    import numpy as np

    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    rows = []
    for i, c2w in enumerate(c2ws):
        w2c = np.linalg.inv(c2w)[:3]
        rows.append([float(i), f / w, f / h, 0.5, 0.5, 0.0, 0.0, *w2c.reshape(-1).tolist()])
    return rows


def sample_size(img_w, img_h, short):
    """(height, width), multiples of 32, keeping the photo's aspect ratio."""
    if img_w >= img_h:
        h = short
        w = int(round(img_w * short / img_h / 32)) * 32
    else:
        w = short
        h = int(round(img_h * short / img_w / 32)) * 32
    return max(32, h), max(32, w)


# ----------------------------------------------------------------------------- model
def stub_triton():
    """videox_fun imports a Triton kernel module (sparse-linear attention) at import time. It is
    never used here and Triton does not exist on Windows; a fake 'triton' module would confuse
    diffusers/transformers (they probe for it), so the kernel module itself is replaced."""
    name = "videox_fun.models.attention_kernel"
    if name in sys.modules:
        return
    try:
        import triton  # noqa: F401  - real Triton (Linux): nothing to do
        return
    except Exception:  # noqa: BLE001
        pass

    def _unavailable(*a, **k):
        raise RuntimeError("sparse-linear attention needs Triton, which is not available on this system")

    mod = types.ModuleType(name)
    mod._sparse_linear_attention = _unavailable
    mod.get_block_map = _unavailable
    sys.modules[name] = mod


def load_pipeline(model_dir, mode, torch):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan3_8, AutoTokenizer, Wan2_2Transformer3DModel, WanT5EncoderModel
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils import apply_gpu_memory_mode, filter_kwargs

    cfg = OmegaConf.create(CONFIG)
    dtype = torch.bfloat16
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        model_dir, transformer_additional_kwargs=OmegaConf.to_container(cfg["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype)
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(model_dir, cfg["vae_kwargs"]["vae_subpath"]),
        additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, cfg["text_encoder_kwargs"]["tokenizer_subpath"]))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_dir, cfg["text_encoder_kwargs"]["text_encoder_subpath"]),
        additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]), low_cpu_mem_usage=True,
        torch_dtype=dtype).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg["scheduler_kwargs"])))
    pipe = Wan2_2FunControlPipeline(transformer=transformer, transformer_2=None, vae=vae, tokenizer=tokenizer,
                                    text_encoder=text_encoder, scheduler=scheduler)
    apply_gpu_memory_mode(pipe, mode, torch.device("cuda"), dtype)
    return pipe


def generate_shot(pipe, image_path, c2ws, hfov, size, steps, seed, prompt, negative, torch, label):
    from videox_fun.data import process_pose_params
    from videox_fun.utils import get_image_to_video_latent

    h, w = size
    n = len(c2ws)
    n = (n - 1) // 4 * 4 + 1
    video, mask, _ = get_image_to_video_latent(image_path, None, video_length=n, sample_size=[h, w])
    cam = process_pose_params(pose_rows(c2ws[:n], hfov, w, h), width=w, height=h, original_pose_width=w,
                              original_pose_height=h)
    cam = cam[:n].permute([3, 0, 1, 2]).unsqueeze(0)
    t0 = [time.time()]

    def on_step(p, i, t, kw):
        now = time.time()
        dt, t0[0] = now - t0[0], now
        used = torch.cuda.memory_reserved() / 2**30
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        progress((i + 1) / steps, f"{label}: step {i + 1}/{steps} · {dt:.1f} s/step · GPU memory {used:.1f}/{tot:.0f} GB")
        return {}

    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        out = pipe(prompt, num_frames=n, negative_prompt=negative, height=h, width=w, generator=gen,
                   guidance_scale=6.0, num_inference_steps=steps, video=video, mask_video=mask, control_video=None,
                   control_camera_video=cam, ref_image=None, boundary=0.875, shift=5,
                   callback_on_step_end=on_step).videos
    return out[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy()   # (F, H, W, 3)


def main(req):
    import json

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="Generating views needs an NVIDIA GPU (CUDA).")
        sys.exit(1)
    kind = req.get("trajectory", "arc")
    if kind not in TRAJECTORIES:
        emit("error", code="bad_input", message=f"unknown trajectory {kind}")
        sys.exit(1)
    os.environ.setdefault("VIDEOX_ATTENTION_TYPE", "SDPA")   # no flash/sage attention on Windows
    stub_triton()
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with Image.open(req["image"]) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    hfov = float(req.get("hfov_deg") or 60.0)
    n = int(req.get("frames", 49))
    steps = int(req.get("steps", 50))
    plan = shots(kind, n)
    torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.92)))
    attempts = [("model_cpu_offload", int(req.get("short_side", 704))),
                ("model_cpu_offload_and_qfloat8", int(req.get("short_side", 704))),
                ("model_cpu_offload_and_qfloat8", 544), ("sequential_cpu_offload", 480)]
    pipe, mode = None, None
    views = [{"path": req["image"], "generated": False}]
    every = max(1, int(req.get("every", 2)))
    for si, c2ws in enumerate(plan):
        for ai, (m, short) in enumerate(attempts):
            try:
                if pipe is None or mode != m:
                    pipe = None
                    torch.cuda.empty_cache()
                    progress(0.0, f"loading Wan 2.2 Fun 5B ({m})")
                    pipe, mode = load_pipeline(req["model_dir"], m, torch), m
                size = sample_size(image.width, image.height, short)
                log(f"shot {si + 1}/{len(plan)}: {size[1]}x{size[0]}, {len(c2ws)} frames, {m}")
                frames = generate_shot(pipe, req["image"], c2ws, hfov, size, steps, int(req.get("seed", 42)) + si,
                                       req.get("prompt") or PROMPT, req.get("negative_prompt") or NEGATIVE, torch,
                                       f"shot {si + 1}/{len(plan)}")
                attempts = attempts[ai:]          # keep the setting that worked for the next shots
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() or ai == len(attempts) - 1:
                    raise
                log(f"out of GPU memory ({m}, {short} px); trying a lighter setting")
                pipe = None
                import gc

                gc.collect()
                torch.cuda.empty_cache()
        for fi in range(every, len(frames), every):      # frame 0 is the photo itself
            p = os.path.join(out_dir, f"shot{si}_{fi:03d}.png")
            Image.fromarray((frames[fi] * 255).round().astype(np.uint8)).save(p)
            views.append({"path": p, "generated": True, "shot": si, "frame": fi})
    with open(os.path.join(out_dir, "views.json"), "w") as f:
        json.dump(views, f)
    emit("result", views=views, vram_peak_mib=vram_peak_mib(torch), trajectory=kind, mode=mode)


if __name__ == "__main__":
    run(main)
