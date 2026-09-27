"""Browser test of the bundled WebXR splat viewer (desktop controls + locomotion maths).

Needs Playwright, a Chromium/Chrome binary and an X display with OpenGL (xvfb-run);
SwiftShader cannot run the splat shader, so it is skipped elsewhere."""

import math
import os
import shutil
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
if not os.environ.get("DISPLAY"):
    pytest.skip("needs an X display with OpenGL (xvfb-run)", allow_module_level=True)


def _chrome() -> str | None:
    for c in (os.environ.get("CHROME_PATH"), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
              shutil.which("google-chrome"), shutil.which("chromium")):
        if c and Path(c).exists():
            return c
    return None


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply
    from twod2vr180.vr_server import VRViewerServer, scene_depth

    import numpy as np

    exe = _chrome()
    if not exe:
        pytest.skip("no Chromium/Chrome binary")
    h, w, f = 120, 160, 150.0
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 3.0)
    z[40:80, 50:110] = 1.5
    pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    img = np.stack([xs * 255 // w, ys * 255 // h, np.full_like(xs, 60)], -1).astype(np.uint8)
    sc = pointmap_to_gaussians(pts, img, None, Camera(w, h, f, f, w / 2, h / 2))
    ply = write_gaussian_ply(sc, tmp_path_factory.mktemp("xr") / "scene.ply")
    srv = VRViewerServer()
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, headless=False,
                              args=["--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": 800, "height": 500})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(srv.share(ply, scene_depth(sc), "test"))
        pg.wait_for_function("window.viewer && window.viewer.splatRenderReady", timeout=60000)
        pg.errors = errors
        yield pg
        b.close()
    srv.stop()


def test_page_loads_without_errors_and_offers_vr(page):
    assert page.errors == []
    buttons = page.evaluate("Array.from(document.querySelectorAll('button')).map(b => b.textContent)")
    assert any("VR" in t for t in buttons)


def test_mouse_drag_and_wheel_move_the_camera(page):
    """Regression (user report on rc3): the mouse did nothing because the library disables its
    controls when WebXR is enabled."""
    page.evaluate("document.getElementById('help').style.display = 'none'")
    before = page.evaluate("window.viewer.camera.position.toArray()")
    page.mouse.move(400, 250)
    page.mouse.down()
    for i in range(15):
        page.mouse.move(400 + i * 12, 250 + i * 3)
        time.sleep(0.03)
    page.mouse.up()
    time.sleep(1.0)
    after = page.evaluate("window.viewer.camera.position.toArray()")
    assert math.dist(before, after) > 0.5
    page.mouse.wheel(0, -600)
    time.sleep(1.0)
    zoomed = page.evaluate("window.viewer.camera.position.toArray()")
    assert math.dist(after, zoomed) > 0.05


def test_locomotion_offset_is_the_inverse_rig_pose(page):
    o = page.evaluate("window.rigToOffset(new window.viewer.camera.position.constructor(1, 0, -2), Math.PI / 2)")
    # rig = T(1,0,-2)·R(90° about y); inverse = T(R(-90°)·(-1,0,2))·R(-90°) = T(-2,0,-1)·R(-90°)
    assert o["position"]["x"] == pytest.approx(-2, abs=1e-6)
    assert o["position"]["z"] == pytest.approx(-1, abs=1e-6)
    assert o["orientation"]["y"] == pytest.approx(-math.sqrt(0.5), abs=1e-6)
    assert o["orientation"]["w"] == pytest.approx(math.sqrt(0.5), abs=1e-6)
