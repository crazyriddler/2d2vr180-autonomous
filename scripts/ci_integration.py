"""Real-model integration run on a GPU-less CI machine with internet access.

Uses the *packaged* CLI exactly as an end user's machine would:
  1. record Hugging Face metadata (revision, SHA256, size, card licence) for every model;
  2. install a backend runtime with the bundled uv (and run its import smoke test);
  3. download the Real 3D models through the model manager (licence accepted explicitly) and verify them.
Inference itself needs the NVIDIA GPU (Real 3D and splat videos are GPU-only); the owner tests it on the
RTX 4080. Artifacts land in --out.

    python scripts/ci_integration.py --app dist/2D2VR/2d2vr-cli.exe --runtime photo-cu128 --out ci-out
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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--runtime", required=True, choices=["photo-cu128", "render-cu124"])
    ap.add_argument("--out", default="ci-out")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("TWOD2VR_HOME", str(out / "home"))
    log = out / "integration.log"
    results: dict = {"runtime": a.runtime}
    if a.runtime == "photo-cu128":
        results["hf_metadata"] = {k: {kk: v.get(kk) for kk in ("revision", "card_license", "gated")}
                                  for k, v in hf_metadata(out).items()}
    rc = sh([a.app, "runtimes", "install", a.runtime], env, log, timeout=5400)
    results["runtime_install"] = rc == 0
    rt_state = Path(env["TWOD2VR_HOME"]) / "runtimes" / a.runtime / "installed.json"
    if rt_state.exists():
        results["runtime_smoke"] = json.loads(rt_state.read_text()).get("smoke_test")
    ok = rc == 0 and bool((results.get("runtime_smoke") or {}).get("ok", True))
    if ok and a.runtime == "photo-cu128":
        for mid in ("moge-3-vitl", "sharp", "yunet-face"):
            got = sh([a.app, "models", "download", mid, "--accept-license"], env, log) == 0
            got = got and sh([a.app, "models", "verify", mid], env, log) == 0
            results[f"download_{mid}"] = got
            ok &= got
        sh([a.app, "models", "list"], env, log)
        sh([a.app, "backends"], env, log)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
