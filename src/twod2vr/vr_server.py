"""Local WebXR splat player server.

Serves the bundled player (assets/webxr/player.html: .splat scenes and .splv splat videos, its renderer
derived from splaTV, MIT) and explicitly shared scene files on http://127.0.0.1 only. Splat videos are
streamed with HTTP range requests (they can be gigabytes). A browser on this PC
(Chrome/Edge) opens it; with a PC-VR headset connected (Quest Link, Air Link,
Virtual Desktop, SteamVR) the page's ENTER VR button starts an immersive
WebXR session. localhost counts as a secure context, so WebXR is allowed
without certificates. Nothing is exposed to the network.
"""

from __future__ import annotations

import http.server
import mimetypes
import secrets
import threading
import urllib.parse
from pathlib import Path

from .paths import resource_root

TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".ply": "application/octet-stream", ".splat": "application/octet-stream",
         ".splv": "application/octet-stream", ".txt": "text/plain; charset=utf-8"}
PAGE = "player.html"


def webxr_dir() -> Path:
    return resource_root() / "assets" / "webxr"


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "2D2VR"

    def log_message(self, *a):  # keep the console quiet
        pass

    def _target(self) -> Path | None:
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/scene/"):
            return self.server.shared.get(path[len("/scene/"):])  # type: ignore[attr-defined]
        name = path.lstrip("/") or PAGE
        cand = webxr_dir() / name
        if "/" not in name and "\\" not in name and cand.is_file():
            return cand
        return None

    def do_HEAD(self):  # noqa: N802
        self.do_GET(body=False)

    def do_GET(self, body: bool = True):  # noqa: N802
        target = self._target()
        if target is None or not target.is_file():
            self.send_error(404)
            return
        size = target.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        partial = False
        if rng and rng.startswith("bytes="):
            try:
                a, b = rng[6:].split(",")[0].split("-")
                if a:
                    start, end = int(a), min(int(b) if b else size - 1, size - 1)
                else:                                   # suffix range: the last N bytes
                    start, end = max(size - int(b), 0), size - 1
                partial = True
            except ValueError:
                partial = False
            if partial and (start > end or start >= size):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", TYPES.get(target.suffix.lower(),
                                                   mimetypes.guess_type(target.name)[0] or "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not body:
            return
        try:
            with open(target, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                        # the player cancelled (seek, page closed)


class VRViewerServer:
    def __init__(self, port: int = 0):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.shared = {}  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="vr-viewer", daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def share(self, scene: Path, depth: float | None = None, name: str | None = None,
              transform: dict | None = None, size: float | None = None) -> str:
        """Expose one scene file (.splat, .splv; a .ply is converted to .splat) under an unguessable name;
        return the player URL.

        ``transform`` (from :func:`align.viewer_transform`) levels the scene and puts the
        capture camera at the viewer's eyes; its "subject" (subject_distance) sets the ideal view and the
        desktop orbit's centre; ``size`` is the starting size (1 = as made)."""
        scene = playable(Path(scene)).resolve()
        token = secrets.token_hex(8) + scene.suffix.lower()
        self.httpd.shared[token] = scene  # type: ignore[attr-defined]
        q = {"scene": f"/scene/{token}", "name": name or scene.parent.parent.name}
        if depth:
            q["depth"] = f"{depth:.3f}"
        if transform:
            q.update(share_url_params(transform))
        if size and abs(size - 1.0) > 1e-3:
            q["s"] = f"{size:.4f}"
        return f"http://127.0.0.1:{self.port}/{PAGE}?" + urllib.parse.urlencode(q)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def share_url_params(transform: dict) -> dict:
    """The player's URL parameters for a view (view_params): levelling q/p, orbit target t, subject distance."""
    q = {"q": ",".join(f"{v:.6f}" for v in transform["quaternion"]),
         "p": ",".join(f"{v:.5f}" for v in transform["position"])}
    if transform.get("target"):
        q["t"] = ",".join(f"{v:.5f}" for v in transform["target"])
    if transform.get("subject"):
        q["subj"] = f"{transform['subject']:.3f}"
    return q


_server: VRViewerServer | None = None


def playable(scene: Path) -> Path:
    """The file the player reads: .splat / .splv as they are; a .ply as the .splat next to it when that is
    up to date, else converted into the app's cache."""
    scene = Path(scene)
    if scene.suffix.lower() != ".ply":
        return scene
    side = scene.with_suffix(".splat")
    if side.is_file() and side.stat().st_mtime >= scene.stat().st_mtime:
        return side
    import hashlib

    from .paths import app_paths
    from .scene import load_scene, write_splat

    st = scene.stat()
    key = hashlib.sha1(f"{scene.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    out = app_paths().cache / "player" / f"{key}.splat"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        write_splat(load_scene(scene), tmp)
        tmp.replace(out)
    return out


def get_server() -> VRViewerServer:
    global _server
    if _server is None:
        _server = VRViewerServer()
    return _server


def scene_depth(scene) -> float | None:
    """Median depth in front of the reference camera (initial orbit target)."""
    import numpy as np

    if scene is None or len(scene) == 0:
        return None
    c2w = scene.cameras[0].c2w if scene.cameras else np.eye(4)
    z = ((scene.means.astype(np.float64) - c2w[:3, 3]) @ np.asarray(c2w)[:3, :3])[:, 2]
    z = z[z > 0]
    return float(np.median(z)) if len(z) else None


def subject_distance(scene) -> float | None:
    """How far the main subject is from the camera (rc10, the ideal VR view): the nearer part (30th
    percentile of depth) of what the camera sees in the middle of its picture (the central half of each side),
    so a person in front of a far wall counts, not the wall. The median depth when the middle is empty."""
    import numpy as np

    if scene is None or len(scene) == 0:
        return None
    cam = scene.cameras[0] if scene.cameras else None
    c2w = np.asarray(cam.c2w if cam is not None else np.eye(4), np.float64)
    p = (scene.means.astype(np.float64) - c2w[:3, 3]) @ c2w[:3, :3]
    z = p[:, 2]
    ok = (z > 0) & np.isfinite(z)
    if getattr(scene, "opacity", None) is not None:
        ok &= np.asarray(scene.opacity) > 0.3
    if not ok.any():
        return scene_depth(scene)
    sel = ok
    if cam is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cam.fx * p[:, 0] / z + cam.cx
            v = cam.fy * p[:, 1] / z + cam.cy
        mid = ok & (np.abs(u - cam.width / 2) < cam.width / 4) & (np.abs(v - cam.height / 2) < cam.height / 4)
        if mid.sum() >= 50:
            sel = mid
    return float(np.percentile(z[sel], 30))


def view_params(scene) -> tuple[float | None, dict | None]:
    """(initial orbit depth, levelling transform) for sharing a scene with the viewer."""
    if scene is None or len(scene) == 0:
        return None, None
    from .align import viewer_transform

    import numpy as np

    depth = scene_depth(scene)
    subject = subject_distance(scene) or depth
    try:
        xf = viewer_transform(scene)
        # Desktop orbit target: the subject, straight ahead of the capture camera
        c2w = np.asarray(scene.cameras[0].c2w if scene.cameras else np.eye(4), np.float64)
        look = c2w[:3, 3] + c2w[:3, 2] * (subject or 2.0)
        xf["subject"] = subject
        from .align import quat_xyzw_to_rotmat

        xf["target"] = [float(v) for v in quat_xyzw_to_rotmat(xf["quaternion"]) @ look + np.asarray(xf["position"])]
    except Exception:  # noqa: BLE001 - levelling is a comfort feature, never block viewing
        xf = None
    return depth, xf


# ------------------------------------------------------------------ browser
def _windows_candidates() -> list[Path]:
    import os

    roots = [os.environ.get(k) for k in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "ProgramW6432")]
    rel = [r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe",
           r"BraveSoftware\Brave-Browser\Application\brave.exe"]
    out = [Path(r) / p for p in rel for r in roots if r]
    try:  # installers register their executables under App Paths
        import winreg

        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for exe in ("chrome.exe", "msedge.exe", "brave.exe"):
                try:
                    with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                        out.append(Path(winreg.QueryValue(k, None)))
                except OSError:
                    pass
    except ImportError:
        pass
    return out


def find_webxr_browser(candidates: list[Path] | None = None) -> Path | None:
    """First installed Chromium-based browser (Chrome, Edge, Brave): these support WebXR with
    OpenXR runtimes (Quest Link / Air Link / SteamVR). Firefox does not."""
    import shutil
    import sys

    if candidates is None:
        if sys.platform == "win32":
            candidates = _windows_candidates()
        else:
            candidates = [Path(p) for p in (shutil.which("google-chrome"), shutil.which("chromium"),
                                            shutil.which("microsoft-edge")) if p]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def open_in_browser(url: str) -> str:
    """Open the viewer; returns the browser used ('default' when no Chromium browser was found)."""
    import subprocess
    import sys
    import webbrowser

    exe = find_webxr_browser()
    if exe is not None:
        try:
            subprocess.Popen([str(exe), url], creationflags=0x08000000 if sys.platform == "win32" else 0)
            return exe.stem
        except OSError:
            pass
    webbrowser.open(url)
    return "default"
