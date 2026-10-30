import urllib.error
import urllib.request

import pytest

from test_scene_io import make_scene
from twod2vr.scene import write_gaussian_ply
from twod2vr.vr_server import VRViewerServer, scene_depth, webxr_dir


@pytest.fixture
def server():
    s = VRViewerServer()
    yield s
    s.stop()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_bundled_player_assets_exist():
    for f in ("player.html", "player.js", "LICENSE-splatv.txt"):
        assert (webxr_dir() / f).is_file(), f


def test_serves_player_and_shared_scene(server, tmp_path):
    sc = make_scene(n=50, sh=False)
    ply = write_gaussian_ply(sc, tmp_path / "export" / "scene.ply")
    url = server.share(ply, scene_depth(sc), "job")
    assert url.startswith(f"http://127.0.0.1:{server.port}/player.html?scene=%2Fscene%2F")
    st, ct, body = get(url)
    assert st == 200 and ct.startswith("text/html") and b"ENTER VR" in body
    st, ct, _ = get(f"http://127.0.0.1:{server.port}/player.js")
    assert st == 200 and ct.startswith("text/javascript")  # ES modules require a JS MIME type
    scene_path = urllib.parse.unquote(url.split("scene=")[1].split("&")[0])
    assert scene_path.endswith(".splat")                    # a .ply is played as .splat
    st, _, body = get(f"http://127.0.0.1:{server.port}{scene_path}")
    assert st == 200 and len(body) == 50 * 32


@pytest.mark.parametrize("path", ["/scene/unknown.ply", "/../pyproject.toml", "/..%2F..%2Fpyproject.toml",
                                  "/sub/player.html"])
def test_only_viewer_files_and_shared_scenes_are_served(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"http://127.0.0.1:{server.port}{path}")
    assert e.value.code == 404


def test_server_binds_loopback_only(server):
    assert server.httpd.server_address[0] == "127.0.0.1"


import urllib.parse  # noqa: E402


def test_browser_choice_prefers_installed_chromium(tmp_path):
    """Regression (user report): Edge is not always installed; Chrome must be used when present."""
    from twod2vr.vr_server import find_webxr_browser

    edge = tmp_path / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    chrome = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    chrome.parent.mkdir(parents=True)
    chrome.write_text("")
    assert find_webxr_browser([edge, chrome]) == chrome  # Edge missing -> Chrome
    assert find_webxr_browser([edge]) is None             # nothing -> caller falls back to default browser


def test_subject_distance_is_the_person_not_the_wall_behind():
    """rc10: the ideal VR view and the desktop orbit centre on the subject: a person 1.2 m away in the middle
    of the picture in front of a wall 5 m away gives ~1.2 m (the median depth would say ~5 m)."""
    import numpy as np

    from twod2vr.scene import Camera, GaussianScene
    from twod2vr.vr_server import share_url_params, subject_distance, view_params

    W, H, f = 400, 300, 350.0
    v, u = np.mgrid[0:H, 0:W] + 0.5                                 # one splat per pixel, as SHARP makes them
    z = np.full((H, W), 5.0)
    person = (np.abs(u - W / 2) < 60) & (v > 60)                   # 0.4 m wide, 1.2 m away, seated in the middle
    z[person] = 1.2 + 0.05 * np.sin(u[person] / 9)
    pts = np.stack([(u - W / 2) / f * z, (v - H / 2) / f * z, z], -1).reshape(-1, 3).astype(np.float32)
    n = len(pts)
    sc = GaussianScene(pts, np.full((n, 3), 0.01, np.float32), np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                       np.full(n, 0.9, np.float32), np.full((n, 3), 0.5, np.float32),
                       cameras=[Camera(W, H, f, f, W / 2, H / 2)])
    d = subject_distance(sc)
    assert 1.1 < d < 1.35
    depth, xf = view_params(sc)
    assert depth > 4.0 and abs(xf["subject"] - d) < 1e-9
    assert abs(np.linalg.norm(xf["target"]) - d) < 0.05            # the orbit turns around the person
    q = share_url_params(xf)
    assert abs(float(q["subj"]) - d) < 1e-3
