"""Fill revision / sha256 / size in config/model-manifest.json from the
Hugging Face API (LFS metadata), for entries hosted on huggingface.co.

    python scripts/resolve_models.py --write

Requires network access to huggingface.co (blocked in the reconnaissance
sandbox; run on a normal machine). Non-HF hosts (Apple CDN) are hashed by
downloading once with --download-other.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HF = re.compile(r"https://huggingface.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)")


def api(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "2d2vr180"}), timeout=60) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--download-other", action="store_true")
    a = ap.parse_args()
    path = REPO / "config" / "model-manifest.json"
    man = json.loads(path.read_text())
    for m in man["models"]:
        for f in m.get("files", []):
            mt = HF.match(f["url"])
            if mt:
                repo, rev, fname = mt.groups()
                info = api(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true")
                sib = {s["rfilename"]: s for s in info.get("siblings", [])}.get(fname)
                if not sib:
                    print(f"{m['id']}: {fname} not found in {repo}", file=sys.stderr)
                    continue
                sha = info["sha"]
                f["url"] = f"https://huggingface.co/{repo}/resolve/{sha}/{fname}"
                f["sha256"] = (sib.get("lfs") or {}).get("sha256")
                f["size_bytes"] = (sib.get("lfs") or {}).get("size") or sib.get("size")
                m["revision"] = sha
                card = (info.get("cardData") or {}).get("license")
                print(f"{m['id']}: rev {sha[:10]} size {f['size_bytes']} sha {str(f['sha256'])[:12]} "
                      f"card-license={card}")
            elif a.download_other:
                h = hashlib.sha256()
                n = 0
                with urllib.request.urlopen(f["url"], timeout=120) as r:
                    for b in iter(lambda: r.read(1 << 20), b""):
                        h.update(b)
                        n += len(b)
                f["sha256"], f["size_bytes"] = h.hexdigest(), n
                print(f"{m['id']}: {n} bytes sha {f['sha256'][:12]}")
    if a.write:
        path.write_text(json.dumps(man, indent=2) + "\n")
        print("manifest updated — review license fields manually before committing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
