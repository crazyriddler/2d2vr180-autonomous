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

TRAJECTORIES = ("capture", "capture_full", "arc", "orbit", "explore", "spiral")

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


def about_target(R, radius=2.0):
    """Camera rotated by R about the point (0, 0, radius) in front of the first camera (OpenCV)."""
    import numpy as np

    c = np.array([0.0, 0.0, radius])
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = c - R @ c
    return m


def orbit_c2w(theta, radius=2.0, pitch=0.0):
    """Horizontal orbit (yaw theta) around the subject."""
    return about_target(_yaw_pitch(-theta, pitch), radius)


def crane_c2w(alpha, radius=2.0):
    """Vertical orbit around the subject: alpha < 0 rises above it looking down, alpha > 0 goes below
    looking up (OpenCV y points down)."""
    return about_target(_yaw_pitch(0.0, alpha), radius)


# (label, kind, degrees, key angles): yaw = horizontal orbit (+ = to the right), crane = vertical orbit
# (- = above, looking down). Key angles are the views used by sharp fusion (None: 1/4, 1/2, 3/4, all).
SHOT_PLANS = {
    "arc": [("right", "yaw", 45, None), ("left", "yaw", -45, None), ("from above", "crane", -35, None),
            ("from below", "crane", 25, None)],
    "orbit": [("right", "yaw", 100, None), ("left", "yaw", -100, None), ("from above", "crane", -50, None),
              ("from below", "crane", 30, None)],
    # 360-degree photo capture: 45, 90, 135, 180 and 270 degrees, overhead, from below
    "capture": [("orbit to 180°", "yaw", 180, (45, 90, 135, 180)), ("orbit to 270°", "yaw", -90, (-90,)),
                ("overhead", "crane", -60, (-60,)), ("from below", "crane", 30, (30,))],
    # ... plus the intermediate angles: 225 and 315 degrees, 30 degrees above
    "capture_full": [("orbit to 180°", "yaw", 180, (45, 90, 135, 180)),
                     ("orbit to 225°", "yaw", -135, (-45, -90, -135)),
                     ("overhead", "crane", -60, (-30, -60)), ("from below", "crane", 30, (30,))],
}
KEY_FRACTIONS = (0.25, 0.5, 0.75, 1.0)   # default key views: fractions of each shot's full angle


def ease_curve(n):
    import numpy as np

    t = np.linspace(0, 1, n)
    return 0.5 - 0.5 * np.cos(np.pi * t)            # smooth start and stop


def key_frames(n, fractions=KEY_FRACTIONS):
    """Frame indices where a shot reaches the given fractions of its full angle."""
    import numpy as np

    e = ease_curve(n)
    return sorted({int(np.argmin(np.abs(e - f))) for f in fractions} - {0})


def shot_key_frames(kind, si, n):
    """Key frames of shot si: at its key angles (capture) or at 1/4 ... 1 of its angle."""
    if kind in SHOT_PLANS:
        _, _, deg, keys = SHOT_PLANS[kind][si]
        if keys:
            return key_frames(n, [k / deg for k in keys])
    return key_frames(n)


def shot_labels(kind):
    if kind in SHOT_PLANS:
        return [plan[0] for plan in SHOT_PLANS[kind]]
    if kind == "explore":
        return ["look right", "look left", "look up", "look down"]
    return ["spiral"]


def shots(kind, n):
    """List of shots; each is a list of n camera-to-world matrices starting at the photo (identity)."""
    import numpy as np

    t = np.linspace(0, 1, n)
    ease = ease_curve(n)
    if kind in SHOT_PLANS:
        out = []
        for _, how, deg, _ in SHOT_PLANS[kind]:
            f = orbit_c2w if how == "yaw" else crane_c2w
            out.append([f(math.radians(deg) * e) for e in ease])
        return out
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


def text_embeddings(model_dir, texts, torch):
    """Encode the prompts once with umT5-XXL (11 GB), then release it.

    Inside the pipeline the text encoder would stay resident - in VRAM while the VAE encodes the
    shot (model offload only evicts it when the transformer starts) and in RAM for the whole run.
    It runs here on the still-empty GPU, or on the CPU if that is not possible."""
    import gc

    from videox_fun.models import AutoTokenizer, WanT5EncoderModel

    cfg = CONFIG["text_encoder_kwargs"]
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, cfg["tokenizer_subpath"]))
    enc = WanT5EncoderModel.from_pretrained(
        os.path.join(model_dir, cfg["text_encoder_subpath"]), additional_kwargs=dict(cfg), low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16).eval()

    def encode(enc, dev):
        enc.to(dev)
        out = []
        for text in texts:
            ids = tok([text], padding="max_length", max_length=cfg["text_length"], truncation=True,
                      add_special_tokens=True, return_tensors="pt")
            n = int(ids.attention_mask.gt(0).sum())
            with torch.no_grad():
                e = enc(ids.input_ids.to(dev), attention_mask=ids.attention_mask.to(dev))[0]
            out.append(e[0, :n].to("cpu", torch.bfloat16))
        return out

    embeds = None
    try:
        embeds = encode(enc, "cuda")
    except torch.cuda.OutOfMemoryError:
        pass
    if embeds is None:  # outside the except block, so the failed attempt's tensors can be freed
        log("text encoder does not fit on the GPU; encoding the prompt on the CPU")
        enc.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        embeds = encode(enc, "cpu")
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    return embeds


def cached_embeddings(model_dir, texts, out_dir, torch):
    """text_embeddings(), stored next to the shots so a restarted attempt does not reload umT5-XXL."""
    import hashlib

    path = os.path.join(out_dir, "prompt_embeds.pt")
    key = hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()
    if os.path.exists(path):
        try:
            d = torch.load(path, map_location="cpu", weights_only=True)
            if d.get("key") == key:
                return d["embeds"]
        except Exception:  # noqa: BLE001 - recompute
            pass
    progress(0.0, "encoding the prompt (umT5-XXL)")
    embeds = text_embeddings(model_dir, texts, torch)
    torch.save({"key": key, "embeds": embeds}, path)
    return embeds


def memory_attempts(vram_total_mib, short_side):
    """Memory settings to try, lightest last. Below 20 GB the bf16 transformer (10 GB) plus a shot's
    activations does not fit, so 16 GB cards start with FP8 weights (half the size)."""
    full = [("model_cpu_offload", short_side)] if vram_total_mib >= 20000 else []
    return full + [("model_cpu_offload_and_qfloat8", short_side), ("model_cpu_offload_and_qfloat8", 544),
                   ("sequential_cpu_offload", 480)]


class _EncodedPrompt:
    """Stands in for the text encoder: the pipeline only reads its dtype (prompts are pre-encoded)."""

    def __init__(self, dtype):
        self.dtype = dtype


def load_pipeline(model_dir, mode, torch, embeds):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan3_8, Wan2_2Transformer3DModel
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
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg["scheduler_kwargs"])))
    pipe = Wan2_2FunControlPipeline(transformer=transformer, transformer_2=None, vae=vae, tokenizer=None,
                                    text_encoder=None, scheduler=scheduler)
    pipe.text_encoder = _EncodedPrompt(dtype)
    pos, neg = embeds

    def encode_prompt(*a, device=None, **k):
        dev = device or torch.device("cuda")
        return [pos.to(dev)], [neg.to(dev)]

    pipe.encode_prompt = encode_prompt
    apply_gpu_memory_mode(pipe, mode, torch.device("cuda"), dtype)
    if mode.startswith("model_cpu_offload"):
        # Model offload evicts a model only when the *next* one in the sequence runs, so the VAE
        # (used first, to encode the photo) would sit in VRAM through every denoising step.
        # Its offload hook moves it back to the GPU for the final decode.
        prepare = pipe.prepare_mask_latents

        def prepare_and_evict(*a, **k):
            out = prepare(*a, **k)
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
            return out

        pipe.prepare_mask_latents = prepare_and_evict
    return pipe


def generate_shot(pipe, image_path, c2ws, hfov, size, steps, seed, prompt, torch, label):
    from videox_fun.data import process_pose_params
    from videox_fun.utils import get_image_to_video_latent

    h, w = size
    n = len(c2ws)
    n = (n - 1) // 4 * 4 + 1
    video, mask, _ = get_image_to_video_latent(image_path, None, video_length=n, sample_size=[h, w])
    cam = process_pose_params(pose_rows(c2ws[:n], hfov, w, h), width=w, height=h, original_pose_width=w,
                              original_pose_height=h)
    cam = cam[:n].permute([3, 0, 1, 2]).unsqueeze(0).to(torch.bfloat16)   # halves a ~1.5 GB per-step copy
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
        out = pipe(prompt, num_frames=n, height=h, width=w, generator=gen,
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
    labels = shot_labels(kind)
    torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.92)))
    # One memory setting per process: a failed attempt's memory (VRAM and RAM) is only reliably
    # released when its process exits, so the app restarts the worker with the next attempt.
    # Finished shots and the encoded prompt are kept on disk and reused.
    attempts = memory_attempts(env.get("vram_total_mib") or 16000, int(req.get("short_side", 704)))
    k = int(req.get("attempt", 0))
    if k >= len(attempts):
        emit("error", code="oom", message="Wan 2.2 does not fit in this GPU's memory at any setting.")
        sys.exit(1)
    m, short = attempts[k]
    prompt = req.get("prompt") or PROMPT
    embeds = cached_embeddings(req["model_dir"], [prompt, req.get("negative_prompt") or NEGATIVE], out_dir, torch)
    views = [{"path": req["image"], "generated": False}]
    every = max(1, int(req.get("every", 2)))
    pipe = None
    for si, c2ws in enumerate(plan):
        done = os.path.join(out_dir, f"shot{si}.json")
        if os.path.exists(done):
            with open(done) as f:
                views += json.load(f)
            log(f"shot {si + 1}/{len(plan)} ({labels[si]}): already generated, reused")
            continue
        if pipe is None:
            progress(0.0, f"loading Wan 2.2 Fun 5B ({m}, attempt {k + 1}/{len(attempts)})")
            pipe = load_pipeline(req["model_dir"], m, torch, embeds)
        size = sample_size(image.width, image.height, short)
        log(f"shot {si + 1}/{len(plan)} ({labels[si]}): {size[1]}x{size[0]}, {len(c2ws)} frames, {m}")
        oom = None
        try:
            frames = generate_shot(pipe, req["image"], c2ws, hfov, size, steps, int(req.get("seed", 42)) + si,
                                   prompt, torch, f"shot {si + 1}/{len(plan)} ({labels[si]})")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = str(e).splitlines()[0][:300]
        if oom:
            more = k + 1 < len(attempts)
            emit("error", code="oom_retry" if more else "oom",
                 message=f"out of GPU memory ({m}, {short} px): {oom}" +
                         ("; retrying with a lighter setting" if more else ""))
            sys.exit(1)
        keys = set(shot_key_frames(kind, si, len(frames)))
        shot_views = []
        for fi in sorted(set(range(every, len(frames), every)) | keys):      # frame 0 is the photo itself
            p = os.path.join(out_dir, f"shot{si}_{fi:03d}.png")
            Image.fromarray((frames[fi] * 255).round().astype(np.uint8)).save(p)
            shot_views.append({"path": p, "generated": True, "shot": si, "frame": fi, "key": fi in keys})
        with open(done, "w") as f:
            json.dump(shot_views, f)
        views += shot_views
        frames = None
    mode = m
    with open(os.path.join(out_dir, "views.json"), "w") as f:
        json.dump(views, f)
    emit("result", views=views, vram_peak_mib=vram_peak_mib(torch), trajectory=kind, mode=mode)


if __name__ == "__main__":
    run(main)
