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
