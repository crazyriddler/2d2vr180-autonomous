import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, app_home):
    from twod2vr.settings import Settings

    Settings(setup_completed=True).save(app_home / "settings.json")
    from twod2vr.gui.app import MainWindow, apply_theme

    apply_theme(qapp)
    w = MainWindow()
    w.show()
    for _ in range(5):
        qapp.processEvents()
    yield w
    w.queue.cancel_all()
    w.close()


def _scene_ply(tmp_path):
    from twod2vr.rgbd import pointmap_to_gaussians
    from twod2vr.scene import Camera, write_gaussian_ply

    h, wd, f = 60, 80, 70.0
    ys, xs = np.mgrid[0:h, 0:wd]
    z = np.full((h, wd), 2.0)
    pts = np.stack([(xs + .5 - wd / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    img = np.stack([xs * 3, ys * 4, np.full_like(xs, 100)], -1).astype(np.uint8)
    return write_gaussian_ply(pointmap_to_gaussians(pts, img, None, Camera(wd, h, f, f, wd / 2, h / 2)),
                              tmp_path / "s.ply")


def test_pages_build(window, qapp):
    w = window
    assert w.stack.count() == 6
    for name in w.PAGES:
        w.go(name)
        qapp.processEvents()
    assert len(w.components.cards) >= 6
    assert "Backends" in w.diagnostics.text.toHtml()
    # a fresh install shows the "install components" banner on the Create page
    w.refresh_hardware()
    assert "Real 3D needs its engine" in w.create.banner.text()
    assert not w.create.banner.isHidden()


def test_viewer_and_open_scene(window, qapp, tmp_path):
    w = window
    w.open_scene(_scene_ply(tmp_path))
    assert w.stack.currentIndex() == 2
    frame = w.viewer_page.viewer.render_frame(160, 120)
    assert frame.shape == (120, 160, 3) and frame.std() > 5
    w.viewer_page.viewer.yaw = 20
    assert not np.array_equal(w.viewer_page.viewer.render_frame(160, 120), frame)


def test_settings_persist(window, app_home):
    w = window
    w.create.ipd.setValue(58)
    w.create.photo_detail.setChecked(True)
    w.create.video_fps.setCurrentIndex(w.create.video_fps.findData(15.0))
    w.settings_page.cpu.setChecked(True)
    saved = json.loads((app_home / "settings.json").read_text())
    assert saved["ipd_mm"] == 58 and saved["video_fps"] == 15.0 and saved["allow_cpu"] is True
    assert saved["photo_detail"] is True
    assert w.ctx.allow_cpu is True
    o = w.settings.job_options()
    assert o.ipd_mm == 58 and o.video_fps == 15.0


def test_queue_and_results_with_fake_backend(window, qapp, ctx, rtx4080, photo):
    """Full GUI flow with the test-only fake worker: queue → job → Results page."""
    import time

    w = window
    w.ctx = ctx
    w.queue.ctx = ctx
    w.hw = rtx4080
    w.create.add_files([str(photo)])
    w.create.generate()
    t0 = time.time()
    while w.queue.pending() and time.time() - t0 < 60:
        qapp.processEvents()
        time.sleep(0.05)
    for _ in range(20):
        qapp.processEvents()
    job = w.queue.jobs[-1]
    assert job.state == "succeeded", job.report.get("error")
    w.results.refresh()
    assert w.results.list.count() >= 1
    w.results.list.setCurrentRow(0)
    assert w.results.btn["explore"].isEnabled() and w.results.btn["vr"].isEnabled()
    assert "photo.jpg" in w.results.details.toPlainText()


def test_gpu_view_falls_back_to_cpu_when_opengl_is_unavailable(window, qapp, tmp_path):
    """Offscreen Qt has no OpenGL: the viewer must switch to the CPU preview, not stay black."""
    import time

    w = window
    w.open_scene(_scene_ply(tmp_path))
    t0 = time.time()
    while time.time() - t0 < 1.0:
        qapp.processEvents()
        time.sleep(0.02)
    v = w.viewer_page.viewer
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        assert not v.using_gl and "CPU" in v.info.text()
    assert w.viewer_page.xr_btn.isEnabled()


@pytest.mark.skipif(os.environ.get("QT_QPA_PLATFORM") == "offscreen" or not os.environ.get("DISPLAY"),
                    reason="needs a display with OpenGL (e.g. xvfb-run)")
def test_gpu_view_renders_with_real_opengl(qapp, tmp_path):
    import time

    from twod2vr.gui.viewer import SceneViewer
    from twod2vr.scene import load_scene

    v = SceneViewer()
    v.resize(400, 300)
    v.show()
    v.set_scene(load_scene(_scene_ply(tmp_path)))
    t0 = time.time()
    while time.time() - t0 < 1.0:
        qapp.processEvents()
        time.sleep(0.02)
    assert v.using_gl and v.gl.ok
    img = v.gl.grabFramebuffer()
    bg = (24, 24, 28)
    px = [img.pixelColor(x, y) for x in range(0, 400, 10) for y in range(0, 300, 10)]
    drawn = sum(abs(c.red() - bg[0]) + abs(c.green() - bg[1]) + abs(c.blue() - bg[2]) > 30 for c in px)
    assert drawn > 0.3 * len(px)  # the scene fills a large part of the view
    v.close()


def _wait(qapp, seconds):
    import time

    t0 = time.time()
    while time.time() - t0 < seconds:
        qapp.processEvents()
        time.sleep(0.02)


def test_viewer_plays_a_splat_video(window, qapp, tmp_path):
    from conftest import FPS, FRAMES, make_video

    w = window
    video = make_video(tmp_path / "export" / "scene.splv")
    w.open_scene(video)
    v = w.viewer_page.viewer
    _wait(qapp, 0.6)                               # offscreen: the GPU view gives way to the CPU preview
    assert v.video is not None and v.video.reader.frames == FRAMES and v.timeline.isVisibleTo(w)
    assert "12 frames" in v.info.text() and w.viewer_page.xr_btn.isEnabled()
    v.set_playing(False)
    v.seek(0.5 / FPS)
    _wait(qapp, 0.5)
    assert v.frame == 0
    def red_x():                                   # the red square in the frame shown
        d = v.frame_data
        return d[d[:, 4] < 0.3, 0].mean()

    x0 = red_x()
    f0 = v.render_frame(160, 120)
    v.seek((FRAMES - 0.5) / FPS)
    _wait(qapp, 0.8)
    assert v.frame == FRAMES - 1
    assert red_x() - x0 > 1.5                      # moved 2 m right
    if not v.using_gl:                             # the CPU preview follows the video too
        assert not np.array_equal(v.render_frame(160, 120), f0)
    # plays at the video's speed (12 frames at 6 fps = 2 s) and loops
    v.seek(0.0)
    v.set_playing(True)
    _wait(qapp, 1.0)
    assert 3 <= v.frame <= 9
    # it reaches the end and loops (a slow machine waits for frames instead of skipping: allow time)
    import time

    seen_end, looped, t0 = False, False, time.time()
    while time.time() - t0 < 10 and not looped:
        _wait(qapp, 0.05)
        seen_end |= v.frame >= FRAMES - 2
        looped = seen_end and v.frame < 6
    assert seen_end and looped
    w.open_scene(_scene_ply(tmp_path))             # a still scene stops the video and hides the timeline
    assert v.video is None and not v.timeline.isVisibleTo(w)
