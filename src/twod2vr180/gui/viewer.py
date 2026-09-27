"""Interactive scene viewer (orbit / zoom / pan) on the reference renderer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ..render import orbit_c2w, render_pinhole
from ..scene import GaussianScene

MAX_PREVIEW_SPLATS = 400_000
PROV_COLORS = np.array([[60, 200, 90], [70, 140, 255], [255, 80, 200], [200, 200, 200]], np.float32) / 255


class SceneViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.scene: GaussianScene | None = None
        self.view_scene: GaussianScene | None = None
        self.yaw = 0.0
        self.pitch = 0.0
        self.dist = 1.0
        self.base_dist = 1.0
        self.center = np.zeros(3)
        self.pan = np.zeros(3)
        self._drag = None
        v = QVBoxLayout(self)
        self.image = QLabel("Open or generate a scene to explore it here.\n"
                            "Drag: orbit · Right-drag: pan · Wheel: zoom")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image.setMinimumSize(320, 240)
        v.addWidget(self.image, 1)
        row = QHBoxLayout()
        self.prov = QCheckBox("Colour by provenance (green observed · blue inferred · pink generative)")
        self.prov.toggled.connect(self._prov_toggled)
        row.addWidget(self.prov)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        row.addWidget(reset)
        self.info = QLabel("")
        row.addWidget(self.info, 1)
        v.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render)
        self.last_frame: np.ndarray | None = None

    def set_scene(self, scene: GaussianScene) -> None:
        self.scene = scene
        if len(scene) > MAX_PREVIEW_SPLATS:
            idx = np.random.default_rng(0).choice(len(scene), MAX_PREVIEW_SPLATS, replace=False)
            sub = scene.subset(np.sort(idx))
            sub.scales = sub.scales * (len(scene) / MAX_PREVIEW_SPLATS) ** (1 / 2)
            self.view_scene = sub
        else:
            self.view_scene = scene
        self._colors = self.view_scene.colors.copy()
        cam = scene.cameras[0] if scene.cameras else None
        base = cam.c2w if cam is not None else np.eye(4)
        self.base = np.asarray(base, np.float64)
        rel = (self.view_scene.means - self.base[:3, 3]) @ self.base[:3, :3]
        depth = float(np.median(rel[:, 2])) if len(rel) else 1.0
        if not np.isfinite(depth) or depth <= 0:
            depth = float(np.linalg.norm(np.median(rel, 0))) or 1.0
        self.base_dist = depth
        self.center = self.base[:3, 3] + self.base[:3, 2] * depth
        self.info.setText(f"{len(scene):,} splats · coverage {scene.coverage()}")
        self.reset_view()

    def reset_view(self) -> None:
        self.yaw = self.pitch = 0.0
        self.dist = self.base_dist
        self.pan = np.zeros(3)
        self.schedule()

    def _prov_toggled(self, on: bool) -> None:
        if self.view_scene is None:
            return
        self.view_scene.colors = PROV_COLORS[np.minimum(self.view_scene.provenance, 3)] * 0.6 + \
            self._colors * 0.4 if on else self._colors.copy()
        self.schedule()

    def schedule(self) -> None:
        self._timer.start(15)

    def render_frame(self, width: int, height: int) -> np.ndarray | None:
        if self.view_scene is None:
            return None
        c2w = orbit_c2w(self.center + self.pan, self.dist, self.yaw, self.pitch, self.base)
        cam = self.scene.cameras[0] if self.scene.cameras else None
        hfov = np.radians(cam.hfov_deg if cam else 60.0)
        f = (width / 2) / np.tan(hfov / 2)
        r = render_pinhole(self.view_scene, c2w, width, height, f, f, width / 2, height / 2,
                           background=(24, 24, 28))
        return r.rgb

    def _render(self) -> None:
        w = max(64, min(self.image.width(), 960))
        h = max(48, min(self.image.height(), 720))
        rgb = self.render_frame(w, h)
        if rgb is None:
            return
        self.last_frame = rgb
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(img))

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, ev):  # noqa: N802
        self._drag = (ev.position().x(), ev.position().y(), ev.button())

    def mouseReleaseEvent(self, ev):  # noqa: N802
        self._drag = None

    def mouseMoveEvent(self, ev):  # noqa: N802
        if not self._drag:
            return
        x0, y0, btn = self._drag
        dx, dy = ev.position().x() - x0, ev.position().y() - y0
        self._drag = (ev.position().x(), ev.position().y(), btn)
        if btn == Qt.RightButton:
            s = self.dist * 0.002
            self.pan += self.base[:3, 0] * (-dx * s) + self.base[:3, 1] * (-dy * s)
        else:
            self.yaw = float(np.clip(self.yaw + dx * 0.3, -170, 170))
            self.pitch = float(np.clip(self.pitch - dy * 0.3, -85, 85))
        self.schedule()

    def wheelEvent(self, ev):  # noqa: N802
        steps = ev.angleDelta().y() / 120
        self.dist = float(np.clip(self.dist * (0.9 ** steps), self.base_dist * 0.05, self.base_dist * 5))
        self.schedule()

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self.schedule()
