import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_main_window_and_viewer(qapp, tmp_path):
    from twod2vr180.gui.app import MainWindow
    from twod2vr180.rgbd import pointmap_to_gaussians
    from twod2vr180.scene import Camera, write_gaussian_ply

    w = MainWindow()
    w.show()
    qapp.processEvents()
    assert w.rt_table.rowCount() >= 2 and w.m_table.rowCount() >= 3
    assert "Backends" in w.diag.toHtml()

    h, wd, f = 60, 80, 70.0
    ys, xs = np.mgrid[0:h, 0:wd]
    z = np.full((h, wd), 2.0)
    pts = np.stack([(xs + .5 - wd / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    img = np.stack([xs * 3, ys * 4, np.full_like(xs, 100)], -1).astype(np.uint8)
    ply = write_gaussian_ply(pointmap_to_gaussians(pts, img, None, Camera(wd, h, f, f, wd / 2, h / 2)),
                             tmp_path / "s.ply")
    w.set_input(str(ply))
    frame = w.viewer.render_frame(160, 120)
    assert frame.shape == (120, 160, 3) and frame.std() > 5
    w.viewer.yaw = 20
    assert not np.array_equal(w.viewer.render_frame(160, 120), frame)
    w.set_input(str(tmp_path / "clip.mp4"))
    assert w.input_path.endswith("clip.mp4")
    w.close()
