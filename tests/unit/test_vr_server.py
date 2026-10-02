import urllib.error
import urllib.request

import pytest

from test_scene_io import make_scene
from twod2vr180.scene import write_gaussian_ply
from twod2vr180.vr_server import VRViewerServer, scene_depth, webxr_dir


@pytest.fixture
def server():
    s = VRViewerServer()
    yield s
    s.stop()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_bundled_viewer_assets_exist():
    for f in ("viewer.html", "three.module.js", "gaussian-splats-3d.module.js",
              "LICENSE-three.txt", "LICENSE-gaussian-splats-3d.txt"):
        assert (webxr_dir() / f).is_file(), f


def test_serves_viewer_modules_and_shared_scene(server, tmp_path):
    sc = make_scene(n=50, sh=False)
    ply = write_gaussian_ply(sc, tmp_path / "export" / "scene.ply")
    url = server.share(ply, scene_depth(sc), "job")
    assert url.startswith(f"http://127.0.0.1:{server.port}/viewer.html?scene=%2Fscene%2F")
    st, ct, body = get(url)
    assert st == 200 and ct.startswith("text/html") and b"ENTER VR" in body
    st, ct, _ = get(f"http://127.0.0.1:{server.port}/three.module.js")
    assert st == 200 and ct.startswith("text/javascript")  # ES modules require a JS MIME type
    scene_path = urllib.parse.unquote(url.split("scene=")[1].split("&")[0])
    st, _, body = get(f"http://127.0.0.1:{server.port}{scene_path}")
    assert st == 200 and body == ply.read_bytes()


@pytest.mark.parametrize("path", ["/scene/unknown.ply", "/../pyproject.toml", "/..%2F..%2Fpyproject.toml",
                                  "/sub/viewer.html"])
def test_only_viewer_files_and_shared_scenes_are_served(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"http://127.0.0.1:{server.port}{path}")
    assert e.value.code == 404


def test_server_binds_loopback_only(server):
    assert server.httpd.server_address[0] == "127.0.0.1"


import urllib.parse  # noqa: E402


def test_browser_choice_prefers_installed_chromium(tmp_path):
    """Regression (user report): Edge is not always installed; Chrome must be used when present."""
    from twod2vr180.vr_server import find_webxr_browser

    edge = tmp_path / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    chrome = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    chrome.parent.mkdir(parents=True)
    chrome.write_text("")
    assert find_webxr_browser([edge, chrome]) == chrome  # Edge missing -> Chrome
    assert find_webxr_browser([edge]) is None             # nothing -> caller falls back to default browser
