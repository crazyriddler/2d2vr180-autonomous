"""TEST-ONLY stand-in for workers/video_splat_worker.py: each frame's pixels on a synthetic depth (darker =
farther) as splats, written exactly like the real worker (frames/NNNNNN.bin + .json, video_meta.json,
resumable). FAKE_VIDEO_OOM_AT=k fails with a GPU out-of-memory error after k frames while the people's model
is on (to test the lighter-settings retry); FAKE_WORKER_DELAY slows every frame."""

import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "workers"))
from _protocol import emit, log, progress, run  # noqa: E402


def main(req):
    import numpy as np
    from PIL import Image

    import splv_codec as codec

    emit("env", torch=None, cuda="12.8", cuda_available=True, device="fake")
    out = req["output_dir"]
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)
    calls = os.path.join(out, "fake_requests.json")
    seen = json.load(open(calls)) if os.path.exists(calls) else []
    json.dump(seen + [req], open(calls, "w"))
    meta_path = os.path.join(out, "video_meta.json")
    oom_at = int(os.environ.get("FAKE_VIDEO_OOM_AT", "-1"))
    delay = float(os.environ.get("FAKE_WORKER_DELAY", "0"))
    frames = req["frames"]
    w, h = Image.open(frames[0]).size
    f = 0.9 * w
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {
        "fov_x": float(np.degrees(2 * np.arctan(w / 2 / f))), "scale": 1.0, "scale_from": "measured depth"}
    meta.update(quant=codec.default_quant(meta["fov_x"], float(np.degrees(2 * np.arctan(h / 2 / f)))),
                camera={"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2},
                count=w * h, grid=req["grid"])
    json.dump(meta, open(meta_path, "w"))
    prev, first, made = None, True, 0
    for i, p in enumerate(frames):
        path = os.path.join(fdir, f"{i:06d}.bin")
        if os.path.exists(path) and os.path.exists(path + ".json"):
            first = True
            continue
        if req.get("people", True) and 0 <= oom_at <= made:
            emit("error", code="oom", message="CUDA out of memory (simulated)")
            sys.exit(1)
        time.sleep(delay)
        img = np.asarray(Image.open(p).convert("RGB"))
        z = 2.0 + (1.0 - img.mean(-1) / 255.0)
        ys, xs = np.mgrid[0:h, 0:w]
        means = np.stack([(xs + 0.5 - w / 2) / f * z, (ys + 0.5 - h / 2) / f * z, z], -1).reshape(-1, 3)
        n = len(means)
        scales = np.repeat((z.reshape(-1, 1) / f) * 0.7, 3, 1)
        quats = np.tile([1.0, 0, 0, 0], (n, 1))
        arr = codec.quantize(means, scales, quats, img.reshape(-1, 3) / 255.0, np.full(n, 0.95), meta["quant"])
        key = first or prev is None or i % int(req.get("keyframe_interval", 15)) == 0
        with open(path, "wb") as fh:
            fh.write(codec.encode_frame(arr, None if key else prev))
        json.dump({"key": bool(key)}, open(path + ".json", "w"))
        prev, first = arr, False
        made += 1
        progress((i + 1) / len(frames), f"frame {i + 1}/{len(frames)}")
    log(f"fake splat video: {len(frames)} frames")
    emit("result", outputs={"frames_dir": fdir, "meta": meta_path}, frames=len(frames), count=w * h,
         camera=meta["camera"], quant=meta["quant"], scale=meta["scale"], scale_from=meta["scale_from"],
         fov_x=meta["fov_x"], vram_peak_mib=None)


if __name__ == "__main__":
    run(main)
