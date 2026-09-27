"""Write release/checksums.sha256 and release/release-manifest.json for the
assets in release/. Records source commit, upstream lock, model and runtime
manifests, bundled tool hashes, build environment and acceptance-gate status.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args) -> str | None:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO)
    return r.stdout.strip() if r.returncode == 0 else None


def main() -> int:
    from twod2vr180 import __version__

    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", default=str(REPO / "release"))
    ap.add_argument("--gates", default=str(REPO / "docs" / "status" / "gates.json"))
    a = ap.parse_args()
    rel = Path(a.release_dir)
    assets = sorted(p for p in rel.iterdir() if p.is_file() and p.name not in ("checksums.sha256",
                                                                             "release-manifest.json"))
    lines = [f"{sha256(p)}  {p.name}" for p in assets]
    (rel / "checksums.sha256").write_text("\n".join(lines) + "\n")
    too_big = [p.name for p in assets if p.stat().st_size >= 2 * 2**30]
    gates = json.loads(Path(a.gates).read_text()) if Path(a.gates).exists() else {}
    manifest = {
        "app": "2D2VR180", "version": __version__,
        "source_commit": git("rev-parse", "HEAD"), "source_dirty": bool(git("status", "--porcelain")),
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_env": {"python": platform.python_version(), "platform": platform.platform()},
        "assets": [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in assets],
        "assets_over_2GiB": too_big,
        "upstream_lock": json.loads((REPO / "config" / "upstream-lock.json").read_text()),
        "model_manifest_sha256": sha256(REPO / "config" / "model-manifest.json"),
        "runtime_manifest": json.loads((REPO / "config" / "runtime-manifest.json").read_text()),
        "bundled_tools": json.loads((REPO / "config" / "build-tools.lock.json").read_text())["tools"],
        "model_weights_bundled": False,
        "acceptance_gates": gates,
    }
    (rel / "release-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n".join(lines))
    if too_big:
        print(f"ERROR: assets >= 2 GiB cannot be GitHub release assets: {too_big}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
