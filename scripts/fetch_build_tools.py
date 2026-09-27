"""Download and verify the third-party executables bundled in the Windows
package (config/build-tools.lock.json) into build/bin/.

Idempotent: a tool already present with a matching wheel hash is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "build" / "bin"))
    ap.add_argument("--cache", default=str(REPO / "build" / "cache"))
    a = ap.parse_args()
    out, cache = Path(a.out), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    lock = json.loads((REPO / "config" / "build-tools.lock.json").read_text())
    for t in lock["tools"]:
        whl = cache / t["wheel"]
        if not whl.exists() or sha256(whl) != t["sha256"]:
            print(f"downloading {t['wheel']}")
            tmp = whl.with_suffix(".part")
            urllib.request.urlretrieve(t["url"], tmp)
            got = sha256(tmp)
            if got != t["sha256"]:
                tmp.unlink()
                print(f"SHA256 mismatch for {t['wheel']}: {got}", file=sys.stderr)
                return 1
            tmp.replace(whl)
        with zipfile.ZipFile(whl) as z:
            for member, target in t["extract"].items():
                dst = out / target
                dst.write_bytes(z.read(member))
                print(f"  {target}: {dst.stat().st_size} bytes, sha256 {sha256(dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
