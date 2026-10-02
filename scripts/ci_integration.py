"""Real-model integration run on a GPU-less CI machine with internet access.

Uses the *packaged* CLI exactly as an end user's machine would:
  1. record Hugging Face metadata (revision, SHA256, size, card licence) for every model;
  2. install a backend runtime with the bundled uv;
  3. download models through the model manager (licence accepted explicitly);
  4. run real inference on CPU on a public-domain photo and a synthetic fixed-camera video;
  5. validate outputs and print a summary. Artifacts land in --out.

    python scripts/ci_integration.py --app dist/2D2VR180/2d2vr180-cli.exe --runtime photo-cu128 --out ci-out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HF = re.compile(r"https://huggingface.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)")


def sh(cmd, env, log, timeout=None) -> int:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    t = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            lf.write(line)
            print("   " + line.rstrip()[:300], flush=True)
        rc = p.wait(timeout=timeout)
    print(f"-> exit {rc} in {time.time() - t:.0f}s", flush=True)
    return rc


def hf_metadata(out: Path) -> dict:
    man = json.loads((REPO / "config" / "model-manifest.json").read_text())
    meta = {}
    for m in man["models"]:
        entry = {"files": {}}
        for f in m.get("files", []):
            mt = HF.match(f["url"])
            if not mt:
                continue
            repo, rev, fname = mt.groups()
            try:
                with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true",
                                            timeout=60) as r:
                    info = json.load(r)
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)
                continue
            sib = {s["rfilename"]: s for s in info.get("siblings", [])}.get(fname, {})
            lfs = sib.get("lfs") or {}
            entry["repo"], entry["revision"] = repo, info.get("sha")
            entry["card_license"] = (info.get("cardData") or {}).get("license")
            entry["gated"] = info.get("gated")
            entry["files"][fname] = {"size": lfs.get("size") or sib.get("size"), "sha256": lfs.get("sha256")}
            try:
                with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{info.get('sha')}/README.md",
                                            timeout=60) as r:
                    card = r.read().decode("utf-8", "replace")
                entry["card_license_lines"] = [ln.strip() for ln in card.splitlines()
                                               if "licen" in ln.lower()][:12]
            except Exception as e:  # noqa: BLE001
                entry["card_error"] = str(e)
        meta[m["id"]] = entry
    (out / "model-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def make_inputs(out: Path) -> dict:
    import numpy as np
    from PIL import Image
    from skimage import data

    photo = out / "astronaut.jpg"  # NASA, public domain (bundled with scikit-image)
    Image.fromarray(data.astronaut()).save(photo, quality=95)
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    bg = np.asarray(Image.fromarray(data.astronaut()).resize((320, 320)))
    video = out / "fixed_camera_moving_object.mp4"
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x320",
                             "-r", "12", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                            stdin=subprocess.PIPE)
    for i in range(36):
        f = bg.copy()
        x = 20 + i * 6
        f[200:260, x:x + 50] = [230, 40, 40]
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return {"photo": photo, "video": video}


def runtime_python(env, rid) -> Path:
    base = Path(env["TWOD2VR180_HOME"]) / "runtimes" / rid / "venv"
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_worker(py, script, req, out, env, log, name) -> dict:
    """Run a worker like the app does and return its result event (or the error)."""
    req_path = out / f"{name}_request.json"
    req_path.write_text(json.dumps(req))
    wenv = dict(env, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                HF_HOME=str(Path(env["TWOD2VR180_HOME"]) / "models" / "hf-cache"))
    p = subprocess.run([str(py), str(REPO / "workers" / script), str(req_path)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=wenv, timeout=5400)
    (out / f"{name}.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(p.stdout[-20000:] + p.stderr[-20000:])
    res = {"exit_code": p.returncode}
    for line in p.stdout.splitlines():
        if line.startswith("@@2D2VR180 "):
            ev = json.loads(line[len("@@2D2VR180 "):])
            if ev.get("event") in ("result", "error"):
                res.update(ev)
    print(name, json.dumps({k: v for k, v in res.items() if k not in ("c2ws", "traceback")})[:2000], flush=True)
    if res.get("traceback"):
        print(res["traceback"], flush=True)
    return res


def model_dir(env, mid) -> Path:
    return Path(env["TWOD2VR180_HOME"]) / "models" / mid


def multiview_selftest(a, env, log, out) -> dict:
    """Real VGGT-1B + MoGe-2 camera poses on the CPU (training needs CUDA and is skipped)."""
    import numpy as np
    from PIL import Image
    from skimage import data

    r = {}
    for mid in ("vggt-1b", "moge-2-vitl-normal"):
        r[f"download_{mid}"] = sh([a.app, "models", "download", mid, "--accept-license"], env, log) == 0
    big = np.asarray(Image.fromarray(data.astronaut()).resize((768, 768)))
    views = []
    for i, dx in enumerate((0, 40, 80)):  # three overlapping crops = a sideways camera move
        p = out / f"mv_view_{i}.png"
        Image.fromarray(big[100:580, 60 + dx:700 + dx]).save(p)
        views.append({"path": str(p), "generated": False, "weight": 1.0})
    res = run_worker(runtime_python(env, "recon3d-cu124"), "multiview_worker.py",
                     {"images": views, "output_dir": str(out / "mv_out"), "vggt_dir": str(model_dir(env, "vggt-1b")),
                      "moge_path": str(model_dir(env, "moge-2-vitl-normal") / "model.pt"), "stop_after_poses": True,
                      "max_side": 640}, out, env, log, "multiview_poses")
    c2ws = np.asarray(res.get("c2ws") or np.zeros((0, 4, 4)))
    ok = res.get("event") == "result" and len(c2ws) == 3 and np.allclose(c2ws[0], np.eye(4), atol=1e-4)
    # The crops come from one photo (no parallax), so the motion is equally explained by a
    # camera pan or a sideways move: either way it must go to the right, monotonically.
    yaw = [float(np.degrees(np.arctan2(m[0, 2], m[2, 2]))) for m in c2ws]
    tx = [float(m[0, 3]) for m in c2ws]
    if ok:
        right = [y + 1e3 * t for y, t in zip(yaw, tx)]
        ok = bool(right[2] > right[1] > right[0] - 1e-6) and int(res.get("points") or 0) > 1000
    r["multiview_poses"] = {"ok": bool(ok), "points": res.get("points"), "metric_scale": res.get("metric_scale_factor"),
                            "camera_x": tx, "yaw_deg": yaw, "error": res.get("message")}
    return r


def generative_selftest(a, env, log, out) -> dict:
    """LaMa inpainting for real (CPU) and Stable Virtual Camera import + trajectory check."""
    import numpy as np
    from PIL import Image
    from skimage import data

    r = {"download_big-lama": sh([a.app, "models", "download", "big-lama", "--accept-license"], env, log) == 0}
    img = np.asarray(Image.fromarray(data.astronaut()).resize((256, 256)))
    mask = np.zeros((256, 256), np.uint8)
    mask[100:140, 110:150] = 255
    holed = img.copy()
    holed[mask > 0] = 0
    Image.fromarray(holed).save(out / "lama_in.png")
    Image.fromarray(mask).save(out / "lama_mask.png")
    res = run_worker(runtime_python(env, "gen-cu128"), "inpaint_worker.py",
                     {"model": str(model_dir(env, "big-lama") / "big-lama.pt"),
                      "items": [{"image": str(out / "lama_in.png"), "mask": str(out / "lama_mask.png"),
                                 "out": str(out / "lama_out.png")}]}, out, env, log, "lama")
    ok = res.get("event") == "result" and (out / "lama_out.png").exists()
    err = None
    if ok:
        o = np.asarray(Image.open(out / "lama_out.png")).astype(float)
        err = float(np.abs(o[100:140, 110:150] - img[100:140, 110:150]).mean())
        ok = bool(err < 60 and o[100:140, 110:150].mean() > 20)   # filled with plausible content, not black
    r["lama"] = {"ok": ok, "mean_abs_error_in_hole": err, "error": res.get("message")}
    code = ("import sys, types; sys.path.insert(0, r'%s'); import seva_worker as sw; "
            "sys.modules['gradio'] = types.SimpleNamespace(Progress=object); import seva.eval, seva.model; "
            "c, K = sw.trajectory('orbit', 21, 60.0, 640, 480); e, _ = sw.trajectory('explore', 21, 60.0, 640, 480); "
            "print('SEVA_OK', c.shape, e.shape)" % (REPO / "workers"))
    p = subprocess.run([str(runtime_python(env, "gen-cu128")), "-c", code], capture_output=True, text=True, env=env)
    print(p.stdout[-2000:], p.stderr[-4000:], flush=True)
    r["seva_import"] = {"ok": "SEVA_OK" in p.stdout, "stderr": p.stderr[-1500:]}
    code = ("import sys, os; sys.path.insert(0, r'%s'); import wan_worker as w; w.stub_triton(); "
            "os.environ['VIDEOX_ATTENTION_TYPE'] = 'SDPA'; "
            "from videox_fun.models import AutoencoderKLWan3_8, AutoTokenizer, Wan2_2Transformer3DModel, "
            "WanT5EncoderModel; from videox_fun.pipeline import Wan2_2FunControlPipeline; "
            "from videox_fun.utils import apply_gpu_memory_mode, filter_kwargs, get_image_to_video_latent; "
            "from videox_fun.data import process_pose_params; "
            "h, wd = w.sample_size(1037, 1555, 704); "
            "cam = process_pose_params(w.pose_rows(w.shots('arc', 49)[0], 60, wd, h), width=wd, height=h, "
            "original_pose_width=wd, original_pose_height=h); print('WAN_OK', tuple(cam.shape))" % (REPO / "workers"))
    p = subprocess.run([str(runtime_python(env, "gen-cu128")), "-c", code], capture_output=True, text=True, env=env)
    print(p.stdout[-2000:], p.stderr[-4000:], flush=True)
    r["wan_import"] = {"ok": "WAN_OK (49, 1056, 704, 6)" in p.stdout, "stderr": p.stderr[-1500:]}
    p = subprocess.run([str(runtime_python(env, "gen-cu128")), str(REPO / "scripts" / "wan_assembly_check.py"),
                        str(REPO / "workers")], capture_output=True, text=True, env=env)
    print(p.stdout[-2000:], p.stderr[-4000:], flush=True)
    r["wan_pipeline"] = {"ok": "WAN_PIPELINE_OK" in p.stdout, "stderr": p.stderr[-1500:]}
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--runtime", required=True, choices=["photo-cu128", "recon3d-cu124", "gen-cu128"])
    ap.add_argument("--out", default="ci-out")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("TWOD2VR180_HOME", str(out / "home"))
    log = out / "integration.log"
    results: dict = {"runtime": a.runtime}

    try:  # which prebuilt gsplat CUDA wheels exist for Windows (decides the GPU renderer runtime)
        with urllib.request.urlopen("https://docs.gsplat.studio/whl/gsplat/", timeout=60) as r:
            idx = r.read().decode("utf-8", "replace")
        wins = sorted(set(re.findall(r"gsplat-[^\"<>]*win_amd64\.whl", idx)))
        results["gsplat_windows_wheels"] = wins[-40:]
        print("gsplat prebuilt Windows wheels:\n  " + "\n  ".join(wins[-40:]), flush=True)
    except Exception as e:  # noqa: BLE001
        results["gsplat_windows_wheels"] = f"index unavailable: {e}"
    if a.runtime == "photo-cu128":
        results["hf_metadata"] = {k: {kk: v.get(kk) for kk in ("revision", "card_license", "gated")}
                                  for k, v in hf_metadata(out).items()}
    rc = sh([a.app, "runtimes", "install", a.runtime], env, log, timeout=5400)
    results["runtime_install"] = rc == 0
    rt_state = Path(env["TWOD2VR180_HOME"]) / "runtimes" / a.runtime / "installed.json"
    if rt_state.exists():
        results["runtime_smoke"] = json.loads(rt_state.read_text()).get("smoke_test")
    if rc == 0 and a.runtime in ("recon3d-cu124", "gen-cu128"):
        fn = multiview_selftest if a.runtime == "recon3d-cu124" else generative_selftest
        try:
            results.update(fn(a, env, log, out))
        except Exception as e:  # noqa: BLE001
            results["selftest_error"] = f"{type(e).__name__}: {e}"
    if rc != 0 or a.runtime != "photo-cu128":
        (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
        print(json.dumps(results, indent=2, default=str))
        ok = rc == 0 and not results.get("selftest_error") and all(
            v.get("ok", True) for v in results.values() if isinstance(v, dict) and "ok" in v)
        return 0 if ok else 1

    for mid in ("depth-anything-v2-small", "moge-2-vits-normal", "sharp"):
        results[f"download_{mid}"] = sh([a.app, "models", "download", mid, "--accept-license"], env, log) == 0
        sh([a.app, "models", "verify", mid], env, log)
    sh([a.app, "models", "list"], env, log)
    sh([a.app, "backends"], env, log)

    inputs = make_inputs(out)
    runs = [
        ("photo_depth_anything", ["run", inputs["photo"], "--backend", "depth_anything_v2", "--mode", "fast"]),
        ("photo_moge_fast", ["run", inputs["photo"], "--backend", "moge_rgbd", "--mode", "fast"]),
        ("photo_sharp", ["run", inputs["photo"], "--backend", "sharp"]),
        ("photo_auto_commercial", ["run", inputs["photo"], "--license-profile", "commercial"]),
        ("video_fixed_camera", ["run", inputs["video"], "--mode", "fast", "--layout", "sbs",
                                "--video-eye-resolution", "512"]),
    ]
    ok_all = True
    for name, args in runs:
        summ = out / f"{name}.json"
        rc = sh([a.app, *args, "--allow-cpu", "--eye-resolution", "1024", "--summary-json", summ, "-v"], env, log,
                timeout=3600)
        s = json.loads(summ.read_text()) if summ.exists() else {"status": "no-summary"}
        s["exit_code"] = rc
        results[name] = s
        ok_all &= s.get("status") == "succeeded"
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    import shutil

    for name, _ in runs:  # keep reports + VR180 stills as CI artifacts
        rep_dir = Path((results[name].get("report") or ""))
        if rep_dir.is_dir():
            dst = out / "jobs" / name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rep_dir / "run_report.json", dst / "run_report.json")
            for img in (rep_dir / "export" / "vr180").glob("*.jpg"):
                shutil.copy2(img, dst / img.name)
    print("\n==== SUMMARY ====")
    for name, _ in runs:
        s = results[name]
        v = s.get("validation") or {}
        print(f"{name:<24} {s.get('status'):<10} backend={s.get('backend')} gaussians={s.get('gaussians')} "
              f"psnr={v.get('psnr_db')} err={(s.get('error') or {}).get('message', '')[:200]}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
