"""Browser test of the bundled player (assets/webxr/player.*): a .splat still and a .splv splat video (a still
background and a red square moving frame by frame, with sound) must load without errors, play at the
video's speed, seek, and change size. Runs headless with SwiftShader (the player's shader is plain WebGL2)."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

playwright = pytest.importorskip("playwright.sync_api")

from conftest import FRAMES, FPS, _plane, make_video  # noqa: E402,F401


def _chrome() -> str | None:
    import glob

    for c in (os.environ.get("CHROME_PATH"), *sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome")),
              shutil.which("google-chrome"), shutil.which("chromium")):
        if c and Path(c).exists():
            return c
    return None


def tone(path: Path, seconds: float) -> tuple[bytes, str] | None:
    """A sound the app would put in a splat video (media.extract_audio of a video with a 440 Hz tone)."""
    from twod2vr.media import AUDIO_MIME, extract_audio, find_ffmpeg

    ff = find_ffmpeg()
    if not ff:
        return None
    src = path.with_suffix(".mkv")
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-c:a", "pcm_s16le", str(src)], check=True)
    out = extract_audio(src, path)
    return (out.read_bytes(), AUDIO_MIME[out.suffix]) if out else None


@pytest.fixture(scope="module")
def browser():
    exe = _chrome()
    if not exe:
        pytest.skip("no Chromium/Chrome binary")
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                    "--autoplay-policy=no-user-gesture-required"])
        yield b
        b.close()


@pytest.fixture(scope="module")
def server():
    from twod2vr.vr_server import VRViewerServer

    s = VRViewerServer()
    yield s
    s.stop()


def _open(browser, url):
    pg = browser.new_page(viewport={"width": 640, "height": 400})
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.goto(url)
    return pg


def _red_x(pg) -> float:
    from io import BytesIO

    from PIL import Image

    img = np.asarray(Image.open(BytesIO(pg.screenshot())).convert("RGB")).astype(int)[:330]
    red = (img[..., 0] > 180) & (img[..., 1] < 120) & (img[..., 2] < 120)
    assert red.sum() > 200, "the moving square is not drawn"
    return float(np.nonzero(red)[1].mean())


def _settled_red_x(pg, timeout=8.0) -> float:
    t0, last = time.time(), None
    while True:
        time.sleep(0.4)
        x = _red_x(pg)
        if last is not None and abs(x - last) < 1.0 or time.time() - t0 > timeout:
            return x
        last = x


def test_still_splat_from_a_ply(browser, server, tmp_path):
    from twod2vr.scene import write_gaussian_ply

    sc = _plane(2.0, [1, 0.1, 0.1], 40, 0.5)
    ply = write_gaussian_ply(sc, tmp_path / "export" / "scene.ply")
    url = server.share(ply, 2.0, "still")
    assert "/player.html?" in url and ".splat" in url          # a .ply is shared as .splat
    pg = _open(browser, url)
    try:
        pg.wait_for_function("window.player && window.player.stats().drawCount === 1600", timeout=60000)
        pg.evaluate("document.getElementById('help').style.display = 'none'")
        _settled_red_x(pg)
        assert pg.errors == [] and pg.evaluate("document.getElementById('err').textContent") == ""
        assert pg.evaluate("player.vid.on") is False
    finally:
        pg.close()


def test_splat_video_plays_seeks_and_has_sound(browser, server, tmp_path):
    audio = tone(tmp_path / "a.m4a", FRAMES / FPS)
    video = make_video(tmp_path / "scene.splv", audio)
    pg = _open(browser, server.share(video, 2.0, "video"))
    try:
        pg.wait_for_function("window.player && window.player.vid.shown === 0", timeout=60000)
        pg.evaluate("document.getElementById('help').style.display = 'none'")
        assert pg.evaluate("player.vid.frames") == FRAMES
        assert pg.evaluate("player.vid.hasAudio") is (audio is not None)
        if audio is not None:                                  # the sound decodes in Chromium (Opus)
            pg.wait_for_function("player.audio.readyState >= 1 || player.audio.error", timeout=20000)
            assert pg.evaluate("player.audio.error") is None
            assert abs(pg.evaluate("player.audio.duration") - FRAMES / FPS) < 0.2
        x0 = _settled_red_x(pg)
        # seek to the last frame: the square is 2 m further right
        pg.evaluate(f"player.seek(({FRAMES - 1} + 0.5) / {FPS})")
        pg.wait_for_function(f"player.vid.shown === {FRAMES - 1}", timeout=30000)
        x1 = _settled_red_x(pg)
        assert x1 - x0 > 150
        # play from the start: frames advance in order at about the video's speed
        pg.evaluate("player.vid.loop = false; player.seek(0); player.setPlaying(true)")
        t0 = time.time()
        seen = set()
        while time.time() - t0 < 15 and pg.evaluate("player.vid.playing"):
            seen.add(pg.evaluate("player.vid.shown"))
            time.sleep(0.05)
        took = time.time() - t0
        assert not pg.evaluate("player.vid.playing")           # reached the end, stopped (no loop)
        assert len(seen) >= FRAMES // 2
        assert took > 0.6 * FRAMES / FPS
        assert pg.errors == [] and pg.evaluate("document.getElementById('err').textContent") == ""
    finally:
        pg.close()


def test_size_buttons_and_keys(browser, server, tmp_path):
    video = make_video(tmp_path / "scene.splv")
    pg = _open(browser, server.share(video, 2.0, "size", size=0.8))
    try:
        pg.wait_for_function("window.player && window.player.vid.shown === 0", timeout=60000)
        assert abs(pg.evaluate("player.size.value") - 0.8) < 1e-6        # starting size from the app
        pg.click("#bigger")
        assert abs(pg.evaluate("player.size.value") - 0.84) < 1e-6
        pg.keyboard.press("0")
        assert pg.evaluate("player.size.value") == 1
        pg.keyboard.press("-")
        assert pg.evaluate("player.size.value") < 1
        assert pg.evaluate("document.getElementById('size').textContent") == "95 %"
    finally:
        pg.close()


FAKE_XR = """
window.XRRigidTransform = class { constructor(p, o) { this.position = p; this.orientation = o; } };
window.fakeXR = (() => {
  const pad = () => ({ axes: [0, 0, 0, 0], buttons: Array.from({ length: 6 }, () => ({ pressed: false })) });
  const left = { handedness: 'left', gamepad: pad() }, right = { handedness: 'right', gamepad: pad() };
  const base = { getOffsetReferenceSpace: (t) => ({ offset: t }) };
  const session = { inputSources: [left, right], requestAnimationFrame: () => 0,
    renderState: { baseLayer: { framebuffer: null, getViewport: () => ({ x: 0, y: 0, width: 1, height: 1 }) } } };
  const head = { pos: [0, 0, 0], fwd: [0, 0, -1] };
  const frame = { session, getViewerPose: (space) => space !== base ? null : { views: [], transform: {
    position: { x: head.pos[0], y: head.pos[1], z: head.pos[2] },
    matrix: [1, 0, 0, 0, 0, 1, 0, 0, -head.fwd[0], -head.fwd[1], -head.fwd[2], 0, 0, 0, 0, 1] } } };
  let t = 0;
  const step = () => { t += 16; player.xrFrame(t, frame); };
  // where the head is in the scene's world, and where it looks (level)
  const world = () => {
    const r = (v, a) => [Math.cos(a) * v[0] + Math.sin(a) * v[2], v[1], -Math.sin(a) * v[0] + Math.cos(a) * v[2]];
    const p = r(head.pos, player.xr.yaw).map((v, i) => v + player.xr.pos[i]);
    const f = r(head.fwd, player.xr.yaw);
    return { p, f };
  };
  return { left, right, base, session, head, step, world };
})();
"""


def test_ideal_view_at_the_start_of_vr_and_on_either_stick_click(browser, server, tmp_path):
    """rc10: entering VR and clicking either thumbstick put the ideal view (the camera's place, a step back when
    the subject is too close, level, facing what the camera faced) where the head is and looks NOW - also after
    walking and turning. The size chosen with A/B stays; A + B together set it back to 100 %."""
    from twod2vr.scene import write_gaussian_ply
    from twod2vr.vr_server import view_params

    sc = _plane(0.5, [1, 0.1, 0.1], 40, 0.3)                    # the subject only 0.5 m away
    ply = write_gaussian_ply(sc, tmp_path / "export" / "scene.ply")
    depth, xf = view_params(sc)
    assert abs(xf["subject"] - 0.5) < 0.02
    pg = _open(browser, server.share(ply, depth, "ideal", xf))
    try:
        pg.wait_for_function("window.player && window.player.stats().drawCount === 1600", timeout=60000)
        eye = pg.evaluate("player.idealEye()")
        assert eye[0] == 0 and eye[1] == 0 and abs(eye[2] - 0.4) < 0.02          # 0.9 m from the subject
        o = pg.evaluate("player.orbit")
        assert abs(o["r"] - 0.9) < 0.03                                          # desktop starts there too
        pg.evaluate(FAKE_XR)
        # enter VR standing 1.6 m up, looking to the right (+x): the ideal view is put there
        pg.evaluate("fakeXR.head.pos = [0.3, 1.6, -0.2]; fakeXR.head.fwd = [1, 0, 0];"
                    "player.startXR(fakeXR.session, fakeXR.base); fakeXR.step()")
        w = pg.evaluate("fakeXR.world()")
        assert np.allclose(w["p"], eye, atol=1e-6) and np.allclose(w["f"], [0, 0, -1], atol=1e-6)
        # walk away and turn; a click of the LEFT stick brings the ideal view back to the head as it is now
        pg.evaluate("fakeXR.head.pos = [2, 1.2, 1]; fakeXR.head.fwd = [-0.6, -0.2, 0.8]; fakeXR.step()")
        assert not np.allclose(pg.evaluate("fakeXR.world()")["p"], eye, atol=1e-3)
        pg.evaluate("player.setSize(2); fakeXR.left.gamepad.buttons[3].pressed = true; fakeXR.step();"
                    "fakeXR.left.gamepad.buttons[3].pressed = false; fakeXR.step()")
        w = pg.evaluate("fakeXR.world()")
        eye2 = pg.evaluate("player.idealEye()")
        assert eye2 == [0, 0, 0]                                     # 1 m away at 200 %: the camera's own place
        assert np.allclose(w["p"], eye2, atol=1e-6)
        assert abs(w["f"][0]) < 1e-6 and w["f"][2] < 0 and abs(w["f"][1] + 0.2) < 1e-6    # faces -z, pitch kept
        assert pg.evaluate("player.size.value") == 2                              # the size stays
        # the RIGHT stick does the same
        pg.evaluate("fakeXR.head.pos = [-1, 1.7, 0]; fakeXR.head.fwd = [0, 0, 1]; fakeXR.step();"
                    "fakeXR.right.gamepad.buttons[3].pressed = true; fakeXR.step();"
                    "fakeXR.right.gamepad.buttons[3].pressed = false; fakeXR.step()")
        w = pg.evaluate("fakeXR.world()")
        assert np.allclose(w["p"], eye2, atol=1e-6) and abs(w["f"][0]) < 1e-6 and w["f"][2] < 0
        # A + B together: original size
        pg.evaluate("const b = fakeXR.right.gamepad.buttons; b[4].pressed = b[5].pressed = true; fakeXR.step();"
                    "b[4].pressed = b[5].pressed = false; fakeXR.step()")
        assert pg.evaluate("player.size.value") == 1
        assert pg.errors == []
    finally:
        pg.close()


def test_range_requests(server, tmp_path):
    import urllib.request

    video = make_video(tmp_path / "scene.splv")
    url = server.share(video)
    path = url.split("scene=")[1].split("&")[0].replace("%2F", "/")
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", headers={"Range": "bytes=4-11"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 206
        assert r.headers["Content-Range"] == f"bytes 4-11/{video.stat().st_size}"
        assert r.read() == video.read_bytes()[4:12]
