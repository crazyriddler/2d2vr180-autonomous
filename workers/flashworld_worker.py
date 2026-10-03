"""Worker (recon3d runtime): one photo -> 3D Gaussians generated directly by FlashWorld.

Stages
  encode   : UMT5-XXL text embedding of "[Static] <prompt>" (bf16, ~11 GB VRAM, run once and cached).
  generate : subject distance and field of view from MoGe-2, a camera path that swings around the
             subject, FlashWorld's 4-step generation (every step decodes 3D Gaussians and renders them
             back into the model, so all views come from one 3D), conversion to the photo's camera frame,
             PLY + turntable frames.

Request (JSON)
  {"stage": "encode"|"generate", "output_dir": "...", "embeds_path": "....pt", "prompt": "",
   "base_dir": "<wan2.2-ti2v-5b-base dir>", "ckpt": "<flashworld model.ckpt>", "moge_path": "...",
   "image": "photo.png", "frames": 24, "short_side": 480, "long_side": 704, "swing_deg": 30,
   "rise_deg": 6, "max_gaussians": 3000000, "min_opacity": 0.01, "offload_vae": false,
   "offload_transformer_during_vae": false}

Coordinates: our frame is the photo's camera in OpenCV convention (x right, y down, z forward), metres.
FlashWorld cameras are OpenGL camera-to-world (x right, y up, looking along -z), normalised so that the
first camera is the origin and the farthest camera is at distance 1 (+0.01)."""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402



def gl_flip():
    import numpy as np

    return np.diag([1.0, -1.0, -1.0, 1.0])


# ----------------------------------------------------------------------------- camera path (testable)
def look_at_cv(eye, target, up=(0.0, -1.0, 0.0)):
    """OpenCV camera-to-world looking from eye to target (y down)."""
    import numpy as np

    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(r) < 1e-9:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, eye
    return m


def swing_path(distance, n=24, swing_deg=30.0, rise_deg=6.0):
    """Camera-to-world matrices (OpenCV, photo camera = identity) swinging around a point `distance`
    in front of the photo: frame 0 is the photo; the camera goes right, back through the photo at
    n/2, left, and back (theta = swing * sin), rising a little at the extremes. The photo's frame is
    first because FlashWorld normalises every camera relative to the first one."""
    import numpy as np

    pivot = np.array([0.0, 0.0, float(distance)])
    out = []
    for k in range(n):
        th = math.radians(swing_deg) * math.sin(2 * math.pi * k / n)
        ph = math.radians(rise_deg) * (math.sin(2 * math.pi * k / n) ** 2)
        # orbit position: rotate the photo camera's offset from the pivot (0, 0, -d) about y, then lift
        off = np.array([-math.sin(th) * math.cos(ph), -math.sin(ph), -math.cos(th) * math.cos(ph)]) * distance
        out.append(look_at_cv(pivot + off, pivot))
    out[0] = np.eye(4)
    return out


def flashworld_cameras(c2ws_cv, fx, fy, cx, cy, width, height):
    """FlashWorld's 11-vector per camera: quaternion (w, x, y, z) and position of the OpenGL
    camera-to-world, then fx/W, fy/H, cx/W, cy/H."""
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multiview_worker import quat_from_rotmat

    rows = []
    for m in c2ws_cv:
        gl = np.asarray(m) @ gl_flip()
        rows.append(list(quat_from_rotmat(gl[:3, :3])) + list(gl[:3, 3]) +
                    [fx / width, fy / height, cx / width, cy / height])
    return np.asarray(rows, np.float32)


def to_photo_frame(xyz, scales, quats, c2w0_cv, t_norm):
    """FlashWorld's Gaussians (normalised frame of its first camera, OpenGL) -> our frame (photo camera,
    OpenCV, metres). normalize_cameras divided positions by (T_norm + 0.01)."""
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multiview_worker import quat_from_rotmat, quat_mul

    s = float(t_norm) + 1e-2
    gl0 = np.asarray(c2w0_cv, np.float64) @ gl_flip()
    R, t = gl0[:3, :3], gl0[:3, 3]
    means = (np.asarray(xyz, np.float64) * s) @ R.T + t
    q = quat_mul(quat_from_rotmat(R), np.asarray(quats, np.float64))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return means.astype(np.float32), (np.asarray(scales, np.float64) * s).astype(np.float32), q.astype(np.float32)


def crop_to(img, width, height):
    """Centre crop to the target aspect and resize; returns (image, scale, crop x0, crop y0) so that a
    pixel (u, v) of the original maps to ((u - x0) * scale, (v - y0) * scale)."""
    from PIL import Image

    h, w = img.shape[:2]
    scale = max(height / h, width / w)
    nw, nh = int(round(width / scale)), int(round(height / scale))
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    import numpy as np

    out = np.asarray(Image.fromarray(img[y0:y0 + nh, x0:x0 + nw]).resize((width, height), Image.LANCZOS))
    return out, width / nw, x0, y0


# ----------------------------------------------------------------------------- stages
def encode(req, torch):
    from transformers import T5TokenizerFast, UMT5EncoderModel

    base = req["base_dir"]
    text = "[Static] " + str(req.get("prompt", ""))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress(0.05, "loading the UMT5-XXL text encoder")
    tok = T5TokenizerFast.from_pretrained(os.path.join(base, "tokenizer"))
    enc = UMT5EncoderModel.from_pretrained(os.path.join(base, "text_encoder"),
                                           torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    enc = enc.to(device).eval()
    ti = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
             return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        e = enc(ti.input_ids.to(device), ti.attention_mask.to(device)).last_hidden_state.float()
    n = int(ti.attention_mask.sum())
    e[:, n:] = 0                       # upstream zero-pads after the prompt's tokens
    os.makedirs(os.path.dirname(os.path.abspath(req["embeds_path"])), exist_ok=True)
    torch.save({"text": text, "embeds": e.cpu()}, req["embeds_path"])
    emit("result", embeds_path=req["embeds_path"], tokens=n, vram_peak_mib=vram_peak_mib(torch))


def subject_distance(moge_path, img, device, torch):
    """(distance to the subject, horizontal field of view in degrees) of the photo from MoGe-2."""
    import numpy as np
    from moge.model.v2 import MoGeModel

    from multiview_worker import subject_mask

    model = MoGeModel.from_pretrained(moge_path).to(device).eval()
    t = torch.from_numpy(np.ascontiguousarray(img)).to(device).float().div(255).permute(2, 0, 1)
    with torch.no_grad():
        res = model.infer(t, resolution_level=6, use_fp16=device == "cuda")
    d = res["depth"].float().cpu().numpy()
    valid = res["mask"].cpu().numpy().astype(bool) if "mask" in res else np.isfinite(d)
    K = res["intrinsics"].float().cpu().numpy() if "intrinsics" in res else None
    del model, res, t
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    m, why = subject_mask(d, valid)
    log(f"photo: {why}")
    h, w = d.shape
    if m is None:          # no separate subject: the middle of the picture
        m = np.zeros_like(valid)
        m[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    sel = m & valid & np.isfinite(d) & (d > 0)
    dist = float(np.median(d[sel])) if sel.any() else 2.0
    hfov = math.degrees(2 * math.atan(0.5 / K[0, 0])) if K is not None else 50.0   # MoGe K is normalised
    return dist, hfov


def generate(req, torch):
    import numpy as np
    from PIL import Image

    import splat_trainer as st
    from multiview_worker import render_turntable

    device = req.get("device", "cuda")      # "cpu" only for tests
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    img = np.asarray(Image.open(req["image"]).convert("RGB"))
    portrait = img.shape[0] > img.shape[1]
    short, long_ = int(req.get("short_side", 480)), int(req.get("long_side", 704))
    W, H = (short, long_) if portrait else (long_, short)
    n = int(req.get("frames", 24))

    progress(0.02, "measuring the subject's distance (MoGe-2)")
    dist, hfov = (subject_distance(req["moge_path"], img, device, torch) if req.get("moge_path")
                  else (float(req.get("distance", 2.0)), 50.0))
    cropped, s, x0, y0 = crop_to(img, W, H)
    fx_full = (img.shape[1] / 2) / math.tan(math.radians(hfov) / 2)
    fx = fy = fx_full * s
    cx, cy = (img.shape[1] / 2 - x0) * s, (img.shape[0] / 2 - y0) * s
    log(f"subject at {dist:.2f} m, field of view {hfov:.0f}°; generating {n} views of {W}x{H}")

    c2ws = swing_path(dist, n, float(req.get("swing_deg", 30)), float(req.get("rise_deg", 6)))
    cams = torch.from_numpy(flashworld_cameras(c2ws, fx, fy, cx, cy, W, H))
    emb = torch.load(req["embeds_path"], map_location="cpu")["embeds"]

    from flashworld.system import GenerationSystem

    base = req["base_dir"]
    with open(os.path.join(base, "transformer", "config.json"), encoding="utf-8") as f:
        tcfg = json.load(f)
    tcfg.update(req.get("transformer_overrides") or {})   # tests: a tiny transformer
    image_t = torch.from_numpy(cropped.copy()).float().permute(2, 0, 1) / 255.0 * 2 - 1
    # Upstream runs the VAE and the 3D decoder on all views at once (~24 GB VRAM). Here they take
    # `frame_chunk` views at a time (identical result) and the FP8 transformer leaves the GPU meanwhile;
    # if memory still runs out, one view at a time. (Upstream's CPU offload of the VAE is far too slow.)
    chunks = [int(req.get("frame_chunk", 4)), 1]
    settings = [(False, True, c) for c in dict.fromkeys(chunks)]
    for attempt, (off_vae, off_tr, chunk) in enumerate(settings):
        oom = False
        progress(0.08, f"loading FlashWorld ({chunk} view{'s' if chunk > 1 else ''} at a time)")
        system = GenerationSystem(os.path.join(base, "vae"), tcfg, os.path.join(base, "scheduler"), req["ckpt"],
                                  device=device, offload_vae=off_vae, offload_transformer_during_vae=off_tr,
                                  log=log, frame_chunk=chunk)
        fg = system.forward_generator
        steps = {"n": 0}

        def logged_step(*a, _fg=fg, **k):     # FlashWorld's 4 steps: the 3D is decoded and re-rendered at each
            steps["n"] += 1
            progress(0.45 + 0.1 * steps["n"], f"generating the 3D scene: step {steps['n']}/4")
            out = _fg(*a, **k)
            if torch.cuda.is_available():
                log(f"step {steps['n']}/4 done (GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
            return out

        system.forward_generator = logged_step
        try:
            progress(0.45, "generating the 3D scene (FlashWorld, 4 steps)")
            with torch.no_grad():
                scene, _, t_norm = system.generate(cams, n, image_t, emb, 0, H, W)
            log("step 4/4 done: 3D Gaussians decoded")
        except torch.cuda.OutOfMemoryError:
            if attempt == len(settings) - 1:
                raise
            oom = True
        del system
        if oom:   # outside the except block: the traceback no longer pins the failed attempt's tensors
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            log("out of GPU memory; retrying with one view at a time")
            continue
        break
    scene = scene.detach().float().cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sh_deg = int(math.sqrt((scene.shape[-1] - 11) / 3) - 1)
    xyz, opacity, scales, rot, feat = scene.split([3, 1, 3, 4, (sh_deg + 1) ** 2 * 3], dim=-1)
    opacity = opacity[:, 0].numpy()
    keep = opacity >= float(req.get("min_opacity", 0.01))
    cap = int(req.get("max_gaussians", 3_000_000))
    if keep.sum() > cap:                       # the most opaque `cap` splats (exactly, ties included)
        idx = np.nonzero(keep)[0]
        top = idx[np.argsort(opacity[idx], kind="stable")[-cap:]]
        keep = np.zeros_like(keep)
        keep[top] = True
    means, sc, q = to_photo_frame(xyz.numpy()[keep], scales.numpy()[keep], rot.numpy()[keep], c2ws[0],
                                  float(t_norm.reshape(-1)[0]))
    dc = feat.reshape(len(feat), (sh_deg + 1) ** 2, 3)[:, 0].numpy()[keep]
    log(f"{int(keep.sum()):,} of {len(opacity):,} Gaussians kept (opacity >= {req.get('min_opacity', 0.01)})")

    progress(0.85, "saving the splat")
    cfg = st.TrainConfig(sh_degree=0)
    params = st.params_from_gaussians({"means": means, "scales": sc, "quats": q, "dc": dc,
                                       "opacity": opacity[keep]}, cfg, device)
    ply = os.path.join(out_dir, "scene.ply")
    st.write_ply(params, ply)
    turntable = None
    try:
        progress(0.9, "rendering a turntable preview")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        turntable = os.path.join(out_dir, "turntable")
        render_turntable(params, K, (W, H), dist, turntable, 0, torch,
                         yaw_deg=0.8 * float(req.get("swing_deg", 30)))
    except Exception as e:  # noqa: BLE001 - a preview only
        log(f"turntable preview failed: {e}")
        turntable = None
    cams_out = [{"width": W, "height": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "c2w": c2ws[0].tolist(),
                 "file": os.path.basename(req["image"]), "generated": False}]
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cams_out, f)
    outputs = {"ply": ply, "cameras": os.path.join(out_dir, "cameras.json")}
    if turntable:
        outputs["turntable_frames"] = turntable
    emit("result", outputs=outputs, gaussians=int(keep.sum()), metric=bool(req.get("moge_path")),
         subject_distance_m=round(dist, 3), hfov_deg=round(hfov, 1), views=n, size=[W, H],
         vram_peak_mib=vram_peak_mib(torch))


def main(req):
    import torch

    env = torch_env(torch)
    stage = req.get("stage", "generate")
    if stage == "encode":
        encode(req, torch)
        return
    if req.get("device", "cuda") == "cuda":
        if not env["cuda_available"]:
            emit("error", code="cuda_unavailable", message="FlashWorld needs an NVIDIA GPU (CUDA).")
            sys.exit(1)
        torch.cuda.set_per_process_memory_fraction(float(req.get("vram_fraction", 0.94)))
    generate(req, torch)


if __name__ == "__main__":
    run(main)
