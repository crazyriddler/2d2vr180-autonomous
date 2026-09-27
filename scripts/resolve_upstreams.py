"""Compare config/upstream-lock.json against the live upstream HEADs.

    python scripts/resolve_upstreams.py          # report drift
    python scripts/resolve_upstreams.py --json   # machine-readable

Never rewrites the lock automatically: moving a pin requires re-running the
backend smoke tests (CLAUDE.md, "Research before integration").
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def head(url: str) -> str | None:
    try:
        out = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.split()[0] if out.returncode == 0 and out.stdout else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    lock = json.loads((REPO / "config" / "upstream-lock.json").read_text())
    rows = []
    for r in lock["repositories"]:
        live = head(r["url"])
        rows.append({"name": r["name"], "pinned": r.get("commit"), "live": live,
                     "status": "unreachable" if live is None else ("ok" if live == r.get("commit") else "drift")})
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        for x in rows:
            print(f"{x['status']:<12} {x['name']:<20} pinned {str(x['pinned'])[:12]}  live {str(x['live'])[:12]}")
    return 1 if any(x["status"] == "unreachable" for x in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
