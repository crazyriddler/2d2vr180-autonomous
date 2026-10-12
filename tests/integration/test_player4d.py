"""Browser test of the bundled 4D player (assets/webxr/player4d.*): a .splatv with a still background
and a square that moves frame by frame must load without errors and show the square where each frame
puts it. Runs headless with SwiftShader (the player's shader is plain WebGL2)."""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

playwright = pytest.importorskip("playwright.sync_api")


def _chrome() -> str | None:
    for c in (os.environ.get("CHROME_PATH"), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
              shutil.which("google-chrome"), shutil.which("chromium")):
        if c and Path(c).exists():
            return c
    return None


def _plane(z, color, n, half, offset=(0.0, 0.0)):
    from twod2vr180.scene import Camera, GaussianScene

    xs, ys = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    m = np.stack([xs.ravel() + offset[0], ys.ravel() + offset[1], np.full(n * n, z)], 1).astype(np.float32)
    k = len(m)
    return GaussianScene(m, np.full((k, 3), 2 * half / n, np.float32), np.tile(np.float32([1, 0, 0, 0]), (k, 1)),
                         np.ones(k, np.float32), np.tile(np.float32(color), (k, 1)),
                         cameras=[Camera(160, 120, 150, 150, 80, 60)])


@pytest.fixture(scope="module")
def player(tmp_path_factory):
    from twod2vr180.scene import Camera
    from twod2vr180.scene4d import DynamicSplats, Scene4D, per_frame_splats
    from twod2vr180.vr_server import VRViewerServer

    exe = _chrome()
    if not exe:
        pytest.skip("no Chromium/Chrome binary")
    frames = 6
    bg = _plane(3.0, [0.2, 0.4, 0.9], 60, 2.0)
    dy = DynamicSplats.concat([per_frame_splats(_plane(1.5, [1, 0.1, 0.1], 20, 0.3, (-1 + 0.4 * k, 0)), k, frames)
                               for k in range(frames)])
    s4 = Scene4D(bg, dy, 6.0, frames, [Camera(160, 120, 150, 150, 80, 60)] * frames, {"playback": "frames"})
    out = s4.save(tmp_path_factory.mktemp("p4d"))
    srv = VRViewerServer()
    url = srv.share(Path(out["splatv"]), 2.0, "test")
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 640, "height": 400})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url)
        pg.wait_for_function("window.player && window.player.play.frames === 6", timeout=60000)
        pg.evaluate("document.getElementById('help').style.display = 'none'")
        pg.errors = errors
        pg.url_used = url
        yield pg
        b.close()
    srv.stop()


def _red_x(pg) -> float:
    from io import BytesIO

    from PIL import Image

    img = np.asarray(Image.open(BytesIO(pg.screenshot())).convert("RGB")).astype(int)[:330]
    red = (img[..., 0] > 180) & (img[..., 1] < 120) & (img[..., 2] < 120)
    assert red.sum() > 200, "the moving square is not drawn"
    return float(np.nonzero(red)[1].mean())


def _settled_red_x(pg, timeout=8.0) -> float:
    """The square's position once the page has re-sorted and redrawn (software GL is slow)."""
    t0, last = time.time(), None
    while True:
        time.sleep(0.4)
        x = _red_x(pg)
        if last is not None and abs(x - last) < 1.0 or time.time() - t0 > timeout:
            return x
        last = x


def test_player_page_is_the_4d_player_and_loads(player):
    assert "player4d.html" in player.url_used
    assert player.errors == []
    assert player.evaluate("document.getElementById('err').textContent") == ""
    assert player.evaluate("player.play.framesMode") is True


def test_each_frame_shows_its_own_splats(player):
    player.keyboard.press(" ")                     # pause
    xs = []
    for k in (0, 5):
        player.evaluate(f"player.play.clock = ({k} + 0.5) / player.play.fps")
        xs.append(_settled_red_x(player))
    assert xs[1] - xs[0] > 150                     # the square moved 2 m to the right


def test_frame_step_keys_and_timeline(player):
    player.evaluate("player.play.clock = 0.5 / player.play.fps")
    player.keyboard.press(".")
    player.keyboard.press(".")
    time.sleep(0.3)
    assert player.evaluate("Math.floor(player.play.clock * player.play.fps)") == 2
    assert "frame 3/6" in player.evaluate("document.getElementById('time').textContent")


def test_moving_gaussians_play_continuously(player, tmp_path):
    """Gaussians with a velocity (smooth 4D motion) are drawn where their motion puts them at any time."""
    from twod2vr180.scene import Camera
    from twod2vr180.scene4d import DynamicSplats, Scene4D, per_frame_splats
    from twod2vr180.vr_server import VRViewerServer

    sq = per_frame_splats(_plane(1.5, [1, 0.1, 0.1], 20, 0.3, (-1.0, 0)), 0, 1)
    sq.t_center[:] = 0.5
    sq.t_scale[:] = 100.0
    sq.motion = np.zeros((len(sq), 9), np.float32)
    sq.motion[:, 0] = 2.0                              # 2 m over the clip: x = -1 + 2 (t - 0.5)
    s4 = Scene4D(_plane(3.0, [0.2, 0.4, 0.9], 60, 2.0), DynamicSplats.concat([sq]), 6.0, 6,
                 [Camera(160, 120, 150, 150, 80, 60)] * 6, {"playback": "continuous"})
    out = s4.save(tmp_path)
    srv = VRViewerServer()
    pg = player.context.browser.new_page(viewport={"width": 640, "height": 400})   # one Playwright session
    try:
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(srv.share(Path(out["splatv"]), 2.0, "motion"))
        pg.wait_for_function("window.player && window.player.play.frames === 6", timeout=60000)
        pg.evaluate("document.getElementById('help').style.display = 'none'")
        assert pg.evaluate("player.play.framesMode") is False
        pg.keyboard.press(" ")
        xs = []
        for t in (0.3, 0.5, 0.7):                        # scene time = clock * fps / frames
            pg.evaluate(f"player.play.clock = {t}")
            xs.append(_settled_red_x(pg))
        assert errors == []
        assert xs[0] < xs[1] < xs[2] and xs[2] - xs[0] > 100
    finally:
        pg.close()
        srv.stop()
