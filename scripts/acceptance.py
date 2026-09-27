"""GPU-free acceptance checks against a built CLI executable.

Covers the Definition-of-done items that do not need a GPU: launch, clear
errors instead of crashes (corrupted input, no GPU, license gate), and a
VR180 render from an existing splat file. GPU items (model download,
inference, video) are in docs/QUALITY_GATES.md and docs/status/gates.json.

    python scripts/acceptance.py --app dist/2D2VR180/2d2vr180-cli[.exe]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    a = ap.parse_args()
    app = str(Path(a.app).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="2d2vr180-acc-"))
    env = dict(os.environ, TWOD2VR180_HOME=str(tmp / "home"))
    results = []

    def run(*args):
        r = subprocess.run([app, *args], capture_output=True, text=True, env=env, timeout=600)
        return r.returncode, r.stdout + r.stderr

    def check(name, ok, detail=""):
        results.append({"check": name, "pass": bool(ok), "detail": detail[-400:]})
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    def report_of(out):
        m = re.search(r"report: (.+run_report\.json)", out)
        return json.loads(Path(m.group(1).strip()).read_text()) if m else None

    rc, out = run("--version")
    check("launch --version", rc == 0 and "2D2VR180" in out, out)
    rc, out = run("doctor", "--json")
    try:
        d = json.loads(out[out.index("{"):])
        check("doctor --json", "gpus" in d and "ffmpeg" in d, out)
        check("bundled ffmpeg found", bool(d.get("ffmpeg")), out)
    except ValueError:
        check("doctor --json", False, out)

    bad = tmp / "corrupt.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    rc, out = run("run", str(bad))
    rep = report_of(out)
    check("corrupted input -> clear error + report", rc == 1 and rep and rep["status"] == "failed"
          and rep["error"]["code"] == "bad_input", out)

    import numpy as np
    from PIL import Image

    img = np.zeros((120, 160, 3), np.uint8)
    img[..., 0] = np.arange(160)[None, :]
    photo = tmp / "photo.png"
    Image.fromarray(img).save(photo)
    rc, out = run("run", str(photo), "--license-profile", "commercial")
    rep = report_of(out)
    check("no usable backend -> explained failure", rc == 1 and rep and rep["error"]["code"] == "no_backend"
          and rep["selection"]["rejected"], out)

    rc, out = run("models", "download", "sharp")
    check("license must be accepted before download", rc == 4, out)

    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    h, w, f = 120, 160, 150.0
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 2.0)
    pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    ply = write_gaussian_ply(pointmap_to_gaussians(pts, img, None, Camera(w, h, f, f, w / 2, h / 2)),
                             tmp / "scene.ply")
    rc, out = run("vr180", str(ply), "--layout", "sbs,tb", "--eye-resolution", "512", "--out", str(tmp / "vr"))
    sbs, tb = tmp / "vr" / "scene_180_LR.jpg", tmp / "vr" / "scene_180_TB.jpg"
    ok = rc == 0 and sbs.exists() and tb.exists()
    if ok:
        ok = Image.open(sbs).size == (1024, 512) and Image.open(tb).size == (512, 1024)
        md = json.loads((tmp / "vr" / "scene_180_LR.json").read_text())
        ok = ok and md["eye_order"] == "left_first" and md["is_full_vr180"] is False
    check("VR180 SBS/TB from splat file", ok, out)

    (tmp / "acceptance.json").write_text(json.dumps(results, indent=2))
    failed = [r for r in results if not r["pass"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed; details in {tmp / 'acceptance.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
