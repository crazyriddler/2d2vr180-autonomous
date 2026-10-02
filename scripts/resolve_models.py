"""Resolve model metadata (exact revision, file sizes, LFS SHA256, licence, gating) from
Hugging Face and GitHub release assets. Runs in CI (the dev container has no HF access);
the JSON it prints is copied into config/model-manifest.json.

    python scripts/resolve_models.py > resolved.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request

HF_REPOS = [
    "alibaba-pai/Wan2.2-Fun-5B-Control-Camera", "alibaba-pai/Wan2.2-Fun-5B-InP",
    "Wan-AI/Wan2.2-TI2V-5B",
]
URLS = [  # non-HF downloads: hashed by streaming
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
]


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "2D2VR180-resolver"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> int:
    out = {"hf": {}, "urls": {}}
    for repo in HF_REPOS:
        try:
            info = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
            files = {}
            for s in info.get("siblings", []):
                lfs = s.get("lfs") or {}
                files[s["rfilename"]] = {"size": lfs.get("size") or s.get("size"), "sha256": lfs.get("sha256")}
            out["hf"][repo] = {"revision": info.get("sha"), "gated": info.get("gated"),
                               "license": (info.get("cardData") or {}).get("license"),
                               "license_name": (info.get("cardData") or {}).get("license_name"),
                               "license_link": (info.get("cardData") or {}).get("license_link"),
                               "files": files}
        except Exception as e:  # noqa: BLE001
            out["hf"][repo] = {"error": f"{type(e).__name__}: {e}"}
    for url in URLS:
        try:
            h = hashlib.sha256()
            n = 0
            req = urllib.request.Request(url, headers={"User-Agent": "2D2VR180-resolver"})
            with urllib.request.urlopen(req, timeout=300) as r:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    h.update(b)
                    n += len(b)
            out["urls"][url] = {"size": n, "sha256": h.hexdigest()}
        except Exception as e:  # noqa: BLE001
            out["urls"][url] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
