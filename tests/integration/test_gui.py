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
    from twod2vr180.settings import Settings

    Settings(setup_completed=True).save(app_home / "settings.json")
    from twod2vr180.gui.app import MainWindow, apply_theme

    apply_theme(qapp)
    w = MainWindow()
    w.show()
    for _ in range(5):
        qapp.processEvents()
    yield w
    w.queue.cancel_all()
    w.close()


def _scene_ply(tmp_path):
    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

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
    assert "photo engine is not installed" in w.create.banner.text()
    assert w.create.backend.count() >= 4


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
    w.settings_page.ipd.setValue(70)
    w.settings_page.cpu.setChecked(True)
    saved = json.loads((app_home / "settings.json").read_text())
    assert saved["eye_separation_mm"] == 70 and saved["allow_cpu"] is True
    assert w.ctx.allow_cpu is True


def test_queue_and_results_with_fake_backend(window, qapp, ctx, rtx4080, photo):
    """Full GUI flow with the test-only fake worker: queue → job → Results page."""
    import time

    w = window
    w.ctx = ctx
    w.queue.ctx = ctx
    w.hw = rtx4080
    w.settings.eye_resolution = 256
    w.settings.still_video_seconds = 0.5
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
    assert w.results.btn["explore"].isEnabled() and w.results.btn["vrvid"].isEnabled()
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

    from twod2vr180.gui.viewer import SceneViewer
    from twod2vr180.scene import load_scene

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


def test_rebuild_groups_candidates_per_angle(tmp_path):
    from twod2vr180.gui.pages import ResultsPage

    gen = tmp_path / "generated_views"
    gen.mkdir()
    for n in ("view00.png", "view00_c1.png", "view01.png", "view01_c1.png", "view01_c2.png", "qwen_prompts.pt"):
        (gen / n).write_bytes(b"x")
    rep = {"backend": {"extra": {"candidate_selection": [
        {"view": "45° left", "chosen": "view00_c1.png"}, {"view": "45° right", "chosen": "view01.png", "dropped": True}]}}}
    groups = ResultsPage._candidate_groups(tmp_path, rep)
    assert [g[0] for g in groups] == ["45° left", "45° right"]
    assert [len(g[1]) for g in groups] == [2, 3]
    assert groups[0][2].endswith("view00_c1.png") and groups[1][2] is None     # dropped → unticked
