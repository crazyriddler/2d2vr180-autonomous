"""Local WebXR splat viewer server.

Serves the bundled viewer (assets/webxr: three.js + GaussianSplats3D, MIT) and
explicitly shared scene files on http://127.0.0.1 only. A browser on this PC
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
         ".txt": "text/plain; charset=utf-8"}


def webxr_dir() -> Path:
    return resource_root() / "assets" / "webxr"


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "2D2VR180"

    def log_message(self, *a):  # keep the console quiet
        pass

    def do_GET(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        target: Path | None = None
        if path.startswith("/scene/"):
            target = self.server.shared.get(path[len("/scene/"):])  # type: ignore[attr-defined]
        else:
            name = path.lstrip("/") or "viewer.html"
            cand = webxr_dir() / name
            if "/" not in name and "\\" not in name and cand.is_file():
                target = cand
        if target is None or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(target.suffix.lower(),
                                                   mimetypes.guess_type(target.name)[0] or "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class VRViewerServer:
    def __init__(self, port: int = 0):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.shared = {}  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="vr-viewer", daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def share(self, scene: Path, depth: float | None = None, name: str | None = None) -> str:
        """Expose one scene file under an unguessable name; return the viewer URL."""
        scene = Path(scene).resolve()
        token = secrets.token_hex(8) + scene.suffix.lower()
        self.httpd.shared[token] = scene  # type: ignore[attr-defined]
        q = {"scene": f"/scene/{token}", "name": name or scene.parent.parent.name}
        if depth:
            q["depth"] = f"{depth:.3f}"
        return f"http://127.0.0.1:{self.port}/viewer.html?" + urllib.parse.urlencode(q)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


_server: VRViewerServer | None = None


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
