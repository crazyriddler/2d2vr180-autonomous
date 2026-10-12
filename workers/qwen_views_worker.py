"""New views of a photo with Qwen-Image-Edit-2511 + the Multiple-Angles LoRA (runtime 'gen-cu128').

An image-editing model (Alibaba Qwen, Apache-2.0) asked to show the same subject from another
camera position: `<sks> [azimuth] [elevation] [distance]` (fal Multiple-Angles LoRA, 96 poses,
Apache-2.0). Each view is generated independently from the user's photo at ~1 megapixel, sharper
than video frames; the Lightning LoRA (lightx2v, Apache-2.0) makes it 4 sampling steps.

Two stages, each in its own process (the app runs them one after the other), so the 7B
vision-language text encoder and the 20B image transformer never share memory:
  stage "encode": Qwen2.5-VL (4-bit) encodes every view prompt together with the photo;
  stage "generate": the GGUF-quantised transformer streams block by block to the GPU.

Request:
  {"stage": "encode" | "generate", "image": "...", "output_dir": "...", "base_dir": "...",
   "gguf": "...", "lora_angles": "...", "lora_lightning": "...",
   "views": [{"label": "90°", "azimuth": 90, "elevation": 0}, ...], "distance": "medium shot",
   "megapixels": 1.0, "steps": 4, "seed": 42, "attempt": 0}

Chained views (rc42): a view may carry "refs" ([{"path", "label"}], views generated in an earlier round) and
"index" (its file number). Its candidates are then drawn from the photo *and* those references (the
nearest accepted view keeps clothes, hair and proportions consistent around the subject); with more than
one candidate, candidate 0 still comes from the photo alone, so the selection can fall back to it.
"""

import gc
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import cap_vram, emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402

AZIMUTHS = {0: "front view", 45: "front-right quarter view", 90: "right side view",
            135: "back-right quarter view", 180: "back view", 225: "back-left quarter view",
            270: "left side view", 315: "front-left quarter view"}
ELEVATIONS = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
DISTANCES = ("close-up", "medium shot", "wide shot")

# Qwen-Image-Lightning scheduler (lightx2v model card): exponential time shift, shift = log(3).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


def nearest(table, value):
    return table[min(table, key=lambda k: min(abs(k - value), 360 - abs(k - value)) if table is AZIMUTHS
                     else abs(k - value))]


def view_prompt(view, distance="medium shot"):
    """'<sks> right side view eye-level shot medium shot' for a view {azimuth, elevation} (degrees;
    azimuth clockwise seen from above, 90 = the subject's right-hand side of the picture). A view
    may carry its own instruction instead ("prompt"), used without the Multiple-Angles LoRA."""
    if view.get("prompt"):
        return view["prompt"]
    az = int(round(view.get("azimuth", 0))) % 360
    return f"<sks> {nearest(AZIMUTHS, az)} {nearest(ELEVATIONS, view.get('elevation', 0))} {distance}"


def output_size(w, h, megapixels):
    """Width/height with the photo's aspect ratio, ~megapixels, multiples of 32."""
    s = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    return max(256, int(round(w * s / 32)) * 32), max(256, int(round(h * s / 32)) * 32)


def system_ram_gib():
    """(total, available) physical memory in GiB, or (None, None) when unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 2**30, st.ullAvailPhys / 2**30
        page = os.sysconf("SC_PAGE_SIZE")
        return os.sysconf("SC_PHYS_PAGES") * page / 2**30, os.sysconf("SC_AVPHYS_PAGES") * page / 2**30
    except (AttributeError, OSError, ValueError):
        return None, None


def memory_attempts(vram_total_mib, megapixels, ram_total_gib=None):
    """(placement, megapixels) to try in order. The Q5 transformer (15 GB) only fits whole on 24 GB+ cards;
    otherwise its blocks stream to the GPU as they are needed ('stream'). Each block is page-locked (pinned)
    on the way; PyTorch keeps freed pinned buffers in a cache that it never returns, which on a 32 GB PC
    exhausted the RAM after two views (rc50 report): the cache is now emptied after every step
    (release_pinned). Group offloading without streams crashed natively on Windows with these GGUF weights
    (rc51 report, access violation at the first step), so it is not used. ``ram_total_gib`` is kept for
    the log only."""
    whole = [("gpu", megapixels)] if vram_total_mib >= 24000 else []
    return whole + [("stream", megapixels), ("stream", min(megapixels, 0.6))]


def release_pinned(torch):
    """Return PyTorch's cached page-locked (pinned) host buffers to Windows. Returns the pinned MB still
    reserved, or None when the stats are unavailable."""
    try:
        torch.cuda.synchronize()
        torch._C._host_emptyCache()
        st = torch.cuda.memory.host_memory_stats()
        return int(st.get("reserved_bytes.all.current", st.get("allocated_bytes.all.current", 0)) // 2**20)
    except Exception:  # noqa: BLE001 - older / other builds: nothing to release
        return None


# ----------------------------------------------------------------------------- stage 1
def encode(req, torch, image):
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (CONDITION_IMAGE_SIZE,
                                                                             calculate_dimensions)
    from diffusers.image_processor import VaeImageProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

    base = req["base_dir"]
    progress(0.0, "loading the Qwen2.5-VL text encoder (4-bit)")
    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    except Exception as e:  # noqa: BLE001 - no bitsandbytes: bf16 split between GPU and RAM (slower)
        log(f"4-bit loading unavailable ({e}); using bf16 with CPU offload")
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "12GiB", "cpu": "30GiB"}
    path = os.path.join(base, "text_encoder")
    try:
        enc = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs).eval()
    except Exception as e:  # noqa: BLE001 - e.g. bitsandbytes cannot load its CUDA library
        if "quantization_config" not in kwargs:
            raise
        log(f"4-bit text encoder failed ({type(e).__name__}: {str(e)[:200]}); using bf16 with CPU offload")
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        enc = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto",
            max_memory={0: "12GiB", "cpu": "30GiB"}).eval()
    processor = Qwen2VLProcessor.from_pretrained(os.path.join(base, "processor"))

    # the same text/image framing as QwenImageEditPlusPipeline._get_qwen_prompt_embeds
    template = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
                "objects, background), then explain how the user's text instruction should alter or modify the "
                "image. Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n")
    drop = 64
    cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
    cond = VaeImageProcessor(vae_scale_factor=16).resize(image, ch, cw)
    from PIL import Image, ImageOps

    def embed(text, images):
        pics = "".join(f"Picture {k + 1}: <|vision_start|><|image_pad|><|vision_end|>" for k in range(len(images)))
        inputs = processor(text=[template.format(pics + text)], images=images, padding=True,
                           return_tensors="pt").to(enc.device)
        with torch.no_grad():
            hs = enc(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                     pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
                     output_hidden_states=True).hidden_states[-1]
        n = int(inputs.attention_mask[0].sum())
        return hs[0, :n][drop:].to("cpu", torch.bfloat16)

    out = []
    views = req["views"]
    for i, v in enumerate(views):
        prompt = view_prompt(v, req.get("distance", "medium shot"))
        item = {"prompt": prompt, "embeds": embed(prompt, [cond])}
        if v.get("refs"):
            refs = []
            for r in v["refs"]:
                with Image.open(r["path"]) as im:
                    ri = ImageOps.exif_transpose(im).convert("RGB")
                rw, rh = calculate_dimensions(CONDITION_IMAGE_SIZE, ri.width / ri.height)
                refs.append(VaeImageProcessor(vae_scale_factor=16).resize(ri, rh, rw))
            names = " and ".join(f"Picture {k + 2} ({r.get('label', 'another view')})" for k, r in enumerate(v["refs"]))
            item["embeds_ref"] = embed(f"{prompt}. The same subject, clothes and hair as in {names}.", [cond] + refs)
        out.append(item)
        progress((i + 1) / len(views), f"encoded view {i + 1}/{len(views)}: {prompt}"
                 + (f" (+ {len(v['refs'])} reference view)" if v.get("refs") else ""))
    torch.save({"key": prompts_key(req), "views": out}, os.path.join(req["output_dir"], "qwen_prompts.pt"))
    emit("result", stage="encode", views=len(out), vram_peak_mib=vram_peak_mib(torch))


def prompts_key(req):
    import hashlib

    refs = [[r["path"], os.path.getsize(r["path"])] for v in req["views"] for r in v.get("refs") or []]
    blob = json.dumps([req["image"], os.path.getsize(req["image"]), req["views"], req.get("distance"), refs],
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- stage 2
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def gguf_state_dict(path):
    """GGUF tensors keyed by diffusers names (ComfyUI-style exports carry a prefix)."""
    from diffusers.models.model_loading_utils import load_gguf_checkpoint

    sd = load_gguf_checkpoint(path)
    out = {}
    for k, v in sd.items():
        if k.startswith("__"):          # ComfyUI markers such as __index_timestep_zero__
            continue
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def load_pipeline(req, torch, placement):
    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                           QwenImageEditPlusPipeline, QwenImageTransformer2DModel)

    base = req["base_dir"]
    progress(0.0, f"loading Qwen-Image-Edit-2511 ({os.path.basename(req['gguf'])}, {placement})")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_state_dict(req["gguf"]), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=base, subfolder="transformer", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLQwenImage.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    pipe = QwenImageEditPlusPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                                     processor=None, transformer=transformer)
    from safetensors.torch import load_file

    # LoRAs are passed as loaded state dicts: with a file path diffusers consults the Hub to guess the
    # weight name, which fails offline (inference never touches the network).
    strength = float(req.get("angles_strength", 0.9))
    if strength > 0:
        pipe.load_lora_weights(load_file(req["lora_angles"]), adapter_name="angles")
    pipe.load_lora_weights(load_file(req["lora_lightning"]), adapter_name="lightning")
    if strength > 0:
        pipe.set_adapters(["angles", "lightning"], adapter_weights=[strength, 1.0])
    else:   # plain instructions (small viewpoint changes): no camera-angle LoRA
        pipe.set_adapters(["lightning"], adapter_weights=[1.0])
    dev = torch.device("cuda")
    if placement == "gpu":
        pipe.to(dev)
    else:
        # Blocks stream from RAM to the GPU as they are needed, pinned on the fly (no second copy of the
        # weights in RAM); the pinned buffers are released after every step (release_pinned).
        transformer.enable_group_offload(onload_device=dev, offload_device=torch.device("cpu"),
                                         offload_type="block_level", num_blocks_per_group=1, use_stream=True,
                                         low_cpu_mem_usage=True)
        vae.to(dev)
    return pipe


def ram_retry(k, attempts, placement, w, h, ex):
    """Out of system memory: retry in a fresh process with the next, lighter setting (finished views are
    kept), or give up with a clear message."""
    more = k + 1 < len(attempts)
    emit("error", code="oom_retry" if more else "oom",
         message=f"out of system memory (RAM) with '{placement}' at {w}x{h}: {type(ex).__name__}: {ex}"
                 + ("; retrying with a lighter setting" if more else
                    ". Close other programs or enlarge the Windows page file and try again."))
    sys.exit(1)


def generate(req, torch, image, env):
    from PIL import Image, ImageOps

    out_dir = req["output_dir"]
    cache = os.path.join(out_dir, "qwen_prompts.pt")
    data = torch.load(cache, map_location="cpu", weights_only=False) if os.path.exists(cache) else None
    if not data or data.get("key") != prompts_key(req):
        emit("error", code="bad_input", message="encoded prompts missing or stale: run the 'encode' stage first")
        sys.exit(1)
    ram_total, ram_free = system_ram_gib()
    attempts = memory_attempts(env.get("vram_total_mib") or 16000, float(req.get("megapixels", 1.0)), ram_total)
    k = int(req.get("attempt", 0))
    if k >= len(attempts):
        emit("error", code="oom", message="Qwen-Image-Edit does not fit in this GPU's memory at any setting.")
        sys.exit(1)
    placement, mp = attempts[k]
    w, h = output_size(image.width, image.height, mp)
    pipe = None
    views = [{"path": req["image"], "generated": False}]
    steps = int(req.get("steps", 4))
    n_cand = max(1, int(req.get("candidates", 1)))
    total = len(req["views"]) * n_cand
    for i, (v, enc) in enumerate(zip(req["views"], data["views"])):
        paths = []
        idx = int(v.get("index", i))
        ref_imgs = []
        for r in v.get("refs") or []:
            with Image.open(r["path"]) as im:
                ref_imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
        for c in range(n_cand):
            # candidate 0 keeps the old file name (results of earlier versions are reused)
            p = os.path.join(out_dir, f"view{idx:02d}.png" if c == 0 else f"view{idx:02d}_c{c}.png")
            use_refs = bool(ref_imgs) and (n_cand == 1 or c >= 1)
            paths.append(p)
            if os.path.exists(p):
                log(f"view {i + 1} ({v.get('label')}) candidate {c + 1}: already generated, reused")
                continue
            if pipe is None:
                if ram_free is not None:
                    log(f"system memory: {ram_free:.1f} of {ram_total:.1f} GB free; transformer placement '{placement}'")
                try:
                    pipe = load_pipeline(req, torch, placement)
                except MemoryError as ex:
                    ram_retry(k, attempts, placement, w, h, ex)
            log(f"view {i + 1}/{len(req['views'])} ({v.get('label')}) candidate {c + 1}/{n_cand}: {enc['prompt']} · "
                f"{w}x{h} · {placement}" + (" · with reference views" if use_refs else ""))
            e = enc["embeds_ref" if use_refs else "embeds"][None].to("cuda", torch.bfloat16)
            mask = torch.ones(e.shape[:2], dtype=torch.long, device="cuda")
            done = i * n_cand + c

            def on_step(pp, s, t, kw, done=done, i=i, c=c):
                pinned = release_pinned(torch)
                if s == 0 and done == 0 and pinned is not None:
                    log(f"pinned host memory after the first step: {pinned} MB (released after every step)")
                progress((done + (s + 1) / steps) / total,
                         f"view {i + 1}/{len(req['views'])} ({v.get('label')}), candidate {c + 1}/{n_cand}: "
                         f"step {s + 1}/{steps}")
                return {}

            oom = None
            try:
                with torch.no_grad():
                    img = pipe(image=[image] + (ref_imgs if use_refs else []), prompt_embeds=e, prompt_embeds_mask=mask, true_cfg_scale=1.0,
                               num_inference_steps=steps, width=w, height=h,
                               generator=torch.Generator("cuda").manual_seed(int(req.get("seed", 42)) + i + 1000 * c),
                               callback_on_step_end=on_step).images[0]
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                if "out of memory" not in str(ex).lower():
                    raise
                oom = str(ex).splitlines()[0][:300]
            except MemoryError as ex:          # system RAM, not the GPU's
                ram_retry(k, attempts, placement, w, h, ex)
            if oom:
                more = k + 1 < len(attempts)
                emit("error", code="oom_retry" if more else "oom",
                     message=f"out of GPU memory ({placement}, {w}x{h}): {oom}" +
                             ("; retrying with a lighter setting" if more else ""))
                sys.exit(1)
            img.save(p)
            del img
            gc.collect()                       # keep RAM flat between candidates
            torch.cuda.empty_cache()
            release_pinned(torch)
        views.append({"path": paths[0], "candidates": paths, "generated": True, "key": True,
                      "label": v.get("label"), "azimuth": v.get("azimuth", 0), "elevation": v.get("elevation", 0),
                      "refs": [r.get("label") for r in v.get("refs") or []]})
    with open(os.path.join(out_dir, "views.json"), "w") as f:
        json.dump(views, f)
    emit("result", stage="generate", views=views, size=[w, h], placement=placement,
         vram_peak_mib=vram_peak_mib(torch))


def main(req):
    import torch
    from PIL import Image, ImageOps

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="Generating views needs an NVIDIA GPU (CUDA).")
        sys.exit(1)
    os.makedirs(req["output_dir"], exist_ok=True)
    with Image.open(req["image"]) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    if max(image.size) > 2048:     # views are ~1 MP: a 24 MP photo only costs memory here
        image.thumbnail((2048, 2048), Image.LANCZOS)
    cap_vram(torch, float(req.get("vram_fraction", 0.92)))
    if req.get("stage") == "encode":
        encode(req, torch, image)
    else:
        generate(req, torch, image, env)


if __name__ == "__main__":
    run(main)
