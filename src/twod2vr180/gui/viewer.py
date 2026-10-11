"""Interactive scene viewer (orbit / pan / zoom / fly) — GPU point-sprite
rendering via OpenGL, with the numpy reference renderer as fallback. 4D scenes
play over time (timeline under the view): the background stays, the moving
splats of each frame replace the previous frame's."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..render import orbit_c2w, render_pinhole
from ..scene import GaussianScene

MAX_CPU_SPLATS = 400_000       # CPU fallback, full quality
MAX_CPU_DRAG_SPLATS = 120_000  # CPU fallback while the mouse is moving
MAX_FRAME_SPLATS = 200_000     # 4D: moving splats shown per frame
PROV_COLORS = np.array([[60, 200, 90], [70, 140, 255], [255, 80, 200], [200, 200, 200]], np.float32) / 255


def _subsample(scene: GaussianScene, n: int, seed: int = 0) -> GaussianScene:
    if len(scene) <= n:
        return scene
    idx = np.sort(np.random.default_rng(seed).choice(len(scene), n, replace=False))
    sub = scene.subset(idx)
    sub.scales = sub.scales * np.sqrt(len(scene) / n)
    return sub


class SceneViewer(QWidget):
    def __init__(self, use_gl: bool = True):
        super().__init__()
        self.scene: GaussianScene | None = None
        self.view_scene: GaussianScene | None = None
        self.drag_scene: GaussianScene | None = None
        self.yaw = 0.0
        self.pitch = 0.0
        self.dist = 1.0
        self.base_dist = 1.0
        self.center = np.zeros(3)
        self.pan = np.zeros(3)
        self.base = np.eye(4)
        self._drag = None
        self._dragging = False
        self.last_frame: np.ndarray | None = None
        v = QVBoxLayout(self)
        self.stack = QStackedWidget()
        self.image = QLabel("Open or generate a scene to explore it here.\n"
                            "Drag: orbit · Right-drag: pan · Wheel: zoom")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image.setMinimumSize(320, 240)
        self.stack.addWidget(self.image)
        self.gl = None
        if use_gl:
            try:
                from .gl_view import GLSplatView

                self.gl = GLSplatView()
                self.gl.failed.connect(self._gl_failed)
                self.gl.setMouseTracking(False)
                self.stack.addWidget(self.gl)
            except Exception:  # noqa: BLE001 - Qt without OpenGL support
                self.gl = None
        v.addWidget(self.stack, 1)
        row = QHBoxLayout()
        self.prov = QCheckBox("Colour by provenance (green observed · blue inferred · pink generative)")
        self.prov.toggled.connect(self._prov_toggled)
        row.addWidget(self.prov)
        row.addWidget(QLabel("Splat size"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(30, 300)
        self.size_slider.setValue(100)
        self.size_slider.setMaximumWidth(140)
        self.size_slider.valueChanged.connect(self._size_changed)
        row.addWidget(self.size_slider)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        row.addWidget(reset)
        self.info = QLabel("")
        self.info.setObjectName("note")
        row.addWidget(self.info, 1)
        v.addLayout(row)
        # 4D timeline (hidden for still scenes)
        self.timeline = QWidget()
        tl = QHBoxLayout(self.timeline)
        tl.setContentsMargins(0, 0, 0, 0)
        self.play_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(lambda: self.set_playing(not self._playing))
        tl.addWidget(self.play_btn)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._slider_moved)
        tl.addWidget(self.frame_slider, 1)
        self.frame_label = QLabel("")
        self.frame_label.setMinimumWidth(150)
        tl.addWidget(self.frame_label)
        self.timeline.setVisible(False)
        v.addWidget(self.timeline)
        self.scene4d = None
        self.frame = 0
        self._moving = None
        self._cpu_frame = None
        self._frame_idx: list[np.ndarray] = []
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render_cpu)
        self._colors: np.ndarray | None = None

    # ------------------------------------------------------------------ state
    @property
    def using_gl(self) -> bool:
        return self.gl is not None and self.stack.currentWidget() is self.gl

    def _gl_failed(self, why: str) -> None:
        self.info.setText(f"GPU view unavailable ({why[:80]}); using CPU preview.")
        self.stack.setCurrentWidget(self.image)
        self.gl = None
        if self.scene4d is not None:
            self.show_frame(self.frame)
        self.schedule()

    def _verify_gl(self) -> None:
        """Qt never calls initializeGL when context creation fails: detect that."""
        if self.gl is not None and self.gl.isVisible() and not (self.gl.isValid() and self.gl.ok):
            self._gl_failed("could not create an OpenGL 3.3 context")

    def set_scene(self, scene: GaussianScene) -> None:
        self._stop_4d()
        self._set_static(scene)

    def _stop_4d(self) -> None:
        self._play_timer.stop()
        self._playing = False
        self.scene4d = None
        self._frame_idx = []
        self.timeline.setVisible(False)

    def set_scene4d(self, s4d) -> None:
        """Play a 4D scene (scene4d.Scene4D): its background is the still scene, the splats visible at each
        frame are shown on top while the timeline plays."""
        self._stop_4d()
        self._set_static(s4d.background)
        self.scene4d = s4d
        dy = s4d.dynamic
        rng = np.random.default_rng(0)
        for k in range(s4d.frames):    # which moving splats belong to each frame (drawn when >= half visible)
            idx = np.nonzero(dy.temporal_opacity(s4d.time_of(k)) >= 0.5)[0]
            if len(idx) > MAX_FRAME_SPLATS:
                idx = np.sort(rng.choice(idx, MAX_FRAME_SPLATS, replace=False))
            self._frame_idx.append(idx)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(s4d.frames - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.timeline.setVisible(s4d.frames > 1)
        self._play_timer.setInterval(int(round(1000 / max(s4d.fps, 1.0))))
        self.show_frame(0)
        self.info.setText(self.info.text() + f" · 4D: {s4d.frames} frames at {s4d.fps:g} fps, "
                          f"{len(dy):,} moving splats")
        self.set_playing(s4d.frames > 1)

    def set_playing(self, on: bool) -> None:
        self._playing = bool(on and self.scene4d is not None)
        self.play_btn.setText("Pause" if self._playing else "Play")
        if self._playing:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _tick(self) -> None:
        if self.scene4d is not None:
            self.show_frame((self.frame + 1) % self.scene4d.frames)

    def _slider_moved(self, k: int) -> None:
        if self.scene4d is not None and k != self.frame:
            self.set_playing(False)
            self.show_frame(k)

    def moving_splats(self, k: int) -> GaussianScene:
        """The moving splats shown at frame ``k`` (moved / faded to that time)."""
        from ..scene4d import DynamicSplats

        dy, idx = self.scene4d.dynamic, self._frame_idx[k]
        pick = lambda a: None if a is None else a[idx]  # noqa: E731
        sub = DynamicSplats(dy.means[idx], dy.scales[idx], dy.quats[idx], dy.opacity[idx], dy.colors[idx],
                            dy.t_center[idx], dy.t_scale[idx], pick(dy.motion), pick(dy.omega), dy.provenance[idx])
        return sub.at(self.scene4d.time_of(k), threshold=0.0)

    def show_frame(self, k: int) -> None:
        if self.scene4d is None:
            return
        self.frame = k
        mv = self.moving_splats(k)
        if self.prov.isChecked():
            mv.colors = (PROV_COLORS[np.minimum(mv.provenance, 3)] * 0.6 + mv.colors * 0.4).astype(np.float32)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(k)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"{k / max(self.scene4d.fps, 1e-6):.2f} s · frame {k + 1}/{self.scene4d.frames}")
        self._moving, self._cpu_frame = mv, None
        if self.gl is not None:
            from .gl_view import splat_radii

            self.gl.set_moving_points(mv.means, mv.colors, splat_radii(mv))
        if not self.using_gl:
            self._sync_cpu_frame()
        self.schedule()

    def _sync_cpu_frame(self) -> None:
        """CPU preview scenes: background + the moving splats of the current frame."""
        from ..scene4d import merge

        if self.scene4d is not None and self._cpu_frame != self.frame:
            self.view_scene = merge(self._bg_view, self._moving)
            self.drag_scene = merge(self._bg_drag, self._moving)
            self._cpu_frame = self.frame

    def _set_static(self, scene: GaussianScene) -> None:
        self.scene = scene
        self.view_scene = self._bg_view = _subsample(scene, MAX_CPU_SPLATS)
        self.drag_scene = self._bg_drag = _subsample(scene, MAX_CPU_DRAG_SPLATS, seed=1)
        self._colors = scene.colors.copy()
        cam = scene.cameras[0] if scene.cameras else None
        self.base = np.asarray(cam.c2w if cam is not None else np.eye(4), np.float64)
        try:  # orbit around a level frame (gravity from the scene geometry)
            from ..align import estimate_up, level_c2w

            self.base = level_c2w(self.base, estimate_up(scene).up)
        except Exception:  # noqa: BLE001 - keep the camera frame
            pass
        rel = (scene.means - self.base[:3, 3]) @ self.base[:3, :3]
        depth = float(np.median(rel[:, 2])) if len(rel) else 1.0
        if not np.isfinite(depth) or depth <= 0:
            depth = float(np.linalg.norm(np.median(rel, 0))) or 1.0
        self.base_dist = depth
        self.center = self.base[:3, 3] + self.base[:3, 2] * depth
        extent = float(np.linalg.norm(np.percentile(scene.means, 99, 0) - np.percentile(scene.means, 1, 0)))
        self._near, self._far = max(depth * 0.005, 1e-3), max(depth * 50, extent * 10, 10.0)
        if self.gl is not None:
            from .gl_view import splat_radii

            self.stack.setCurrentWidget(self.gl)
            self.gl.set_points(scene.means, scene.colors, splat_radii(scene))
            QTimer.singleShot(400, self._verify_gl)
        self.info.setText(f"{len(scene):,} splats · " + ("GPU view" if self.gl is not None else "CPU preview")
                          + " · " + ", ".join(f"{k} {v:.0%}" for k, v in scene.coverage().items() if v))
        self.reset_view()

    def reset_view(self) -> None:
        self.yaw = self.pitch = 0.0
        self.dist = self.base_dist
        self.pan = np.zeros(3)
        self.schedule()

    def _prov_toggled(self, on: bool) -> None:
        if self.scene is None:
            return
        cols = (PROV_COLORS[np.minimum(self.scene.provenance, 3)] * 0.6 + self._colors * 0.4) if on else self._colors
        self.scene.colors = cols.astype(np.float32)
        self.view_scene = self._bg_view = _subsample(self.scene, MAX_CPU_SPLATS)
        self.drag_scene = self._bg_drag = _subsample(self.scene, MAX_CPU_DRAG_SPLATS, seed=1)
        if self.gl is not None:
            from .gl_view import splat_radii

            self.gl.set_points(self.scene.means, self.scene.colors, splat_radii(self.scene))
        if self.scene4d is not None:
            self.show_frame(self.frame)
        self.schedule()

    def _size_changed(self, v: int) -> None:
        if self.gl is not None:
            self.gl.splat_scale = v / 100.0
        self.schedule()

    def camera(self) -> tuple[np.ndarray, float]:
        c2w = orbit_c2w(self.center + self.pan, self.dist, self.yaw, self.pitch, self.base)
        cam = self.scene.cameras[0] if self.scene is not None and self.scene.cameras else None
        vfov = cam.vfov_deg if cam else 45.0
        return c2w, float(np.clip(vfov, 20, 100))

    def schedule(self) -> None:
        if self.scene is None:
            return
        if self.using_gl:
            c2w, vfov = self.camera()
            self.gl.set_camera(c2w, vfov, self._near, self._far)
        else:
            self._timer.start(0 if self._dragging else 15)

    # ------------------------------------------------------------------ CPU fallback
    def render_frame(self, width: int, height: int, fast: bool = False) -> np.ndarray | None:
        if self.scene is None:
            return None
        c2w, vfov = self.camera()
        f = (height / 2) / np.tan(np.radians(vfov) / 2)
        self._sync_cpu_frame()
        sc = self.drag_scene if fast else self.view_scene
        r = render_pinhole(sc, c2w, width, height, f, f, width / 2, height / 2, background=(24, 24, 28),
                           crack_fill=not fast)
        return r.rgb

    def _render_cpu(self) -> None:
        if self.scene is None:
            return
        full_w = max(64, min(self.image.width(), 1280))
        full_h = max(48, min(self.image.height(), 800))
        if self._dragging:  # quarter the pixels and a third of the splats while moving
            w, h = max(64, full_w // 2), max(48, full_h // 2)
        else:
            w, h = full_w, full_h
        rgb = self.render_frame(w, h, fast=self._dragging)
        self.last_frame = rgb
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(img)
        if (w, h) != (full_w, full_h):
            pm = pm.scaled(full_w, full_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.image.setPixmap(pm)

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, ev):  # noqa: N802
        self._drag = (ev.position().x(), ev.position().y(), ev.button())
        self._dragging = True

    def mouseReleaseEvent(self, ev):  # noqa: N802
        self._drag = None
        self._dragging = False
        self.schedule()  # full-quality frame after moving

    def mouseMoveEvent(self, ev):  # noqa: N802
        if not self._drag:
            return
        x0, y0, btn = self._drag
        dx, dy = ev.position().x() - x0, ev.position().y() - y0
        self._drag = (ev.position().x(), ev.position().y(), btn)
        if btn == Qt.RightButton or btn == Qt.MiddleButton:
            c2w, _ = self.camera()
            s = self.dist * 0.0015
            self.pan += c2w[:3, 0] * (-dx * s) + c2w[:3, 1] * (-dy * s)
        else:
            self.yaw = float(np.clip(self.yaw + dx * 0.3, -170, 170))
            self.pitch = float(np.clip(self.pitch - dy * 0.3, -85, 85))
        self.schedule()

    def wheelEvent(self, ev):  # noqa: N802
        steps = ev.angleDelta().y() / 120
        self.dist = float(np.clip(self.dist * (0.9 ** steps), self.base_dist * 0.02, self.base_dist * 5))
        self.schedule()

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self.schedule()
