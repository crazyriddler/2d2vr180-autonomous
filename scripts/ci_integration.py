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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--runtime", required=True, choices=["photo-cu128", "recon3d-cu124"])
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
    if rc != 0 or a.runtime != "photo-cu128":
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return 0 if rc == 0 else 1

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
