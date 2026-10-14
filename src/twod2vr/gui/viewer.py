"""Interactive scene viewer (orbit / pan / zoom / fly) — GPU point-sprite
rendering via OpenGL, with the numpy reference renderer as fallback. Splat videos
(.splv) play with their sound: a thread decodes frames ahead, the sound is the clock
(no drift), and when a frame is not ready yet playback waits instead of skipping."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..render import orbit_c2w, render_pinhole
from ..scene import GaussianScene

MAX_CPU_SPLATS = 400_000       # CPU fallback, full quality
MAX_CPU_DRAG_SPLATS = 120_000  # CPU fallback while the mouse is moving
PROV_COLORS = np.array([[60, 200, 90], [70, 140, 255], [255, 80, 200], [200, 200, 200]], np.float32) / 255


def _subsample(scene: GaussianScene, n: int, seed: int = 0) -> GaussianScene:
    if len(scene) <= n:
        return scene
    idx = np.sort(np.random.default_rng(seed).choice(len(scene), n, replace=False))
    sub = scene.subset(idx)
    sub.scales = sub.scales * np.sqrt(len(scene) / n)
    return sub


class FrameDecoder:
    """Decodes the frames of a splat video ahead of playback, into the GPU view's vertex layout, in two
    stages that run at the same time (zlib and numpy release the GIL): a thread decodes the frames in order
    (each may depend on the previous one), a small pool turns them into vertices. Memory stays bounded: at
    most ``ahead`` frames."""

    def __init__(self, path: Path, ahead_bytes: float = 400e6):
        from concurrent.futures import ThreadPoolExecutor

        from .. import splv

        self.reader = splv.open_reader(path)
        self.fast = splv.codec().FastDequantizer(self.reader.header["quant"])
        n = self.reader.count
        self.ahead = int(min(max(ahead_bytes // max(n * 28, 1), 4), 3 * max(self.reader.fps, 1)))
        self.pool = ThreadPoolExecutor(2, thread_name_prefix="splv-vertices")
        self.cond = threading.Condition()
        self.ready: dict = {}                 # frame -> Future of its vertex array
        self.next = 0
        self.gen = 0
        self.stop_flag = False
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="splv-decoder", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            with self.cond:
                while not self.stop_flag and (self.next >= self.reader.frames or len(self.ready) >= self.ahead):
                    self.cond.wait()
                if self.stop_flag:
                    return
                k, gen = self.next, self.gen
            try:
                q = self.reader.quantized(k)
                fut = self.pool.submit(self.fast.packed, q)
            except Exception as e:  # noqa: BLE001 - shown in the viewer
                with self.cond:
                    self.error = f"{type(e).__name__}: {e}"
                    self.stop_flag = True
                return
            with self.cond:
                if gen == self.gen:
                    self.ready[k] = fut
                    self.next = k + 1
                    self.cond.notify_all()

    def take(self, k: int) -> np.ndarray | None:
        """Frame ``k`` if ready (older frames are dropped). Decoding jumps to ``k`` when the decoder is not on
        its way there."""
        with self.cond:
            for old in [j for j in self.ready if j < k]:
                self.ready.pop(old).cancel()
            fut = self.ready.get(k)
            data = None
            if fut is not None and fut.done():
                del self.ready[k]
                try:
                    data = fut.result()
                except Exception as e:  # noqa: BLE001 - shown in the viewer
                    self.error = f"{type(e).__name__}: {e}"
            elif fut is None and not (self.next <= k <= self.next + self.ahead):
                self.seek(k)
            self.cond.notify_all()
            return data

    def seek(self, k: int) -> None:
        with self.cond:
            self.gen += 1
            for f in self.ready.values():
                f.cancel()
            self.ready.clear()
            self.next = k
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.stop_flag = True
            self.cond.notify_all()
        self.pool.shutdown(wait=False, cancel_futures=True)


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
        # splat video timeline (hidden for still scenes)
        self.timeline = QWidget()
        tl = QHBoxLayout(self.timeline)
        tl.setContentsMargins(0, 0, 0, 0)
        self.play_btn = QPushButton("Play")
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
        self.video = None                    # FrameDecoder of the splat video shown
        self._video_desc = ""
        self.frame = -1
        self._playing = False
        self._waiting = False
        self._clock = 0.0
        self._t0 = 0.0
        self._audio = None
        self._audio_out = None
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(10)
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
        self.info.setText((self._video_desc + " · " if self.video is not None else "")
                          + f"GPU view unavailable ({why[:80]}); using CPU preview.")
        self.stack.setCurrentWidget(self.image)
        self.gl = None
        self.schedule()

    def _verify_gl(self) -> None:
        """Qt never calls initializeGL when context creation fails: detect that."""
        if self.gl is not None and self.gl.isVisible() and not (self.gl.isValid() and self.gl.ok):
            self._gl_failed("could not create an OpenGL 3.3 context")

    def set_scene(self, scene: GaussianScene) -> None:
        self.stop_video()
        self._set_static(scene)

    # ------------------------------------------------------------------ splat video
    def stop_video(self) -> None:
        self._play_timer.stop()
        self._playing = False
        if self.video is not None:
            self.video.close()
        self.video = None
        if self._audio is not None:
            self._audio.stop()
        self._audio = self._audio_out = None
        self.timeline.setVisible(False)

    def set_video(self, path: Path) -> None:
        """Play a splat video (.splv): the first frame sets the view, then the timeline plays it with sound."""
        from .. import splv

        self.stop_video()
        dec = FrameDecoder(Path(path))
        self._set_static(splv.frame_scene(dec.reader, 0))
        self.video = dec
        self.frame = -1
        self._clock = 0.0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(dec.reader.frames - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.timeline.setVisible(dec.reader.frames > 1)
        self._audio_setup(Path(path), dec.reader)
        self._video_desc = (f"splat video: {dec.reader.frames} frames at {dec.reader.fps:g} fps, "
                            f"{dec.reader.count:,} splats per frame" + (", with sound" if self._audio else ""))
        self.info.setText(self._video_desc + (" · GPU view" if self.gl is not None else " · CPU preview (slow)"))
        self._play_timer.start()
        self.set_playing(dec.reader.frames > 1)

    def _audio_setup(self, path: Path, reader) -> None:
        a = reader.header.get("audio")
        if not a:
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except Exception:  # noqa: BLE001 - Qt without multimedia: silent playback
            return
        import hashlib

        from ..paths import app_paths

        st = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
        ext = ".webm" if "webm" in (a.get("mime") or "") else ".m4a"
        f = app_paths().cache / "player" / f"{key}{ext}"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(reader.audio())
        self._audio_out = QAudioOutput(self)
        self._audio = QMediaPlayer(self)
        self._audio.setAudioOutput(self._audio_out)
        self._audio.setSource(QUrl.fromLocalFile(str(f)))

    @property
    def duration(self) -> float:
        return self.video.reader.frames / self.video.reader.fps if self.video else 0.0

    def clock(self) -> float:
        return self._clock + (time.monotonic() - self._t0 if self._playing and not self._waiting else 0.0)

    def set_playing(self, on: bool) -> None:
        if self.video is None:
            return
        if on and not self._playing and self.clock() >= self.duration - 0.5 / self.video.reader.fps:
            self.seek(0.0)
        self._clock = self.clock()
        self._t0 = time.monotonic()
        self._playing = bool(on)
        self.play_btn.setText("Pause" if self._playing else "Play")
        self._sync_audio()

    def _sync_audio(self) -> None:
        if self._audio is None:
            return
        if self._playing and not self._waiting:
            self._audio.setPosition(int(self._clock * 1000))
            self._audio.play()
        else:
            self._audio.pause()

    def seek(self, t: float) -> None:
        if self.video is None:
            return
        fps = self.video.reader.fps
        self._clock = float(min(max(t, 0.0), (self.video.reader.frames - 0.5) / fps))
        self._t0 = time.monotonic()
        self.video.seek(int(self._clock * fps))
        self._sync_audio()

    def _tick(self) -> None:
        dec = self.video
        if dec is None:
            return
        if dec.error:
            self.info.setText(f"Cannot play the splat video: {dec.error}")
            self.stop_video()
            return
        fps, n = dec.reader.fps, dec.reader.frames
        t = self.clock()
        if self._playing and not self._waiting and self._audio is not None:
            from PySide6.QtMultimedia import QMediaPlayer

            if self._audio.playbackState() == QMediaPlayer.PlayingState:   # the sound is the clock
                a = self._audio.position() / 1000.0
                if abs(a - t) > 0.12:
                    self._clock, self._t0, t = a, time.monotonic(), a
        if t >= n / fps:
            if n > 1 and self._playing:
                self.seek(0.0)
                t = 0.0
            else:
                t = (n - 0.5) / fps
        want = min(int(t * fps), n - 1)
        if want != self.frame:
            data = dec.take(want)
            if data is not None:
                self._show_packed(want, data)
                if self._waiting:
                    self._waiting = False
                    self._t0 = time.monotonic()
                    self._sync_audio()
            elif self._playing and not self._waiting:
                self._clock, self._waiting = t, True          # wait for the frame (stable speed, no skips)
                self._sync_audio()
                self.frame_label.setText("buffering…")

    def _show_packed(self, k: int, data: np.ndarray) -> None:
        self.frame = k
        self.frame_data = data                 # positions (3), colours (3), footprint radius (1) per splat
        fps = self.video.reader.fps
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(k)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"{k / fps:.1f} / {self.duration:.1f} s")
        if self.using_gl:
            self.gl.set_packed(data)
        elif not self._playing or k % max(int(fps), 1) == 0:     # CPU preview: about one frame a second
            from .. import splv

            self._update_scene(splv.frame_scene(_reader_copy(self.video), k))
        self.schedule()

    def _update_scene(self, scene: GaussianScene) -> None:
        """Replace the scene shown without moving the view (CPU preview of a video frame)."""
        self.scene = scene
        self.view_scene = self._bg_view = _subsample(scene, MAX_CPU_SPLATS)
        self.drag_scene = self._bg_drag = _subsample(scene, MAX_CPU_DRAG_SPLATS, seed=1)
        self._colors = scene.colors.copy()

    def _slider_moved(self, k: int) -> None:
        if self.video is not None:
            self.set_playing(False)
            self.seek((k + 0.5) / self.video.reader.fps)

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


def _reader_copy(dec: FrameDecoder):
    """A reader of the same file for the GUI thread (the decoder's own reader belongs to its thread)."""
    from .. import splv

    if getattr(dec, "_gui_reader", None) is None:
        dec._gui_reader = splv.open_reader(dec.reader.path)
    return dec._gui_reader
