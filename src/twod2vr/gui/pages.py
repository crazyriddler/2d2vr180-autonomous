"""Application pages: Create, Results, 3D Viewer, Components, Settings, Diagnostics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..media import IMAGE_EXTS, VIDEO_EXTS

MODE_HELP = {
    "auto": "balanced (recommended)",
    "quality": "more detail; videos: more splats per frame, slower",
    "fast": "quicker; videos: fewer splats per frame",
}
PROVENANCE_HELP = ("<b>Observed</b>: reconstructed from several views. <b>Inferred</b>: seen in the input but its "
                   "depth is predicted by a network. <b>Generative</b>: painted by AI where the photo shows nothing "
                   "(Real 3D fill). <b>Unknown</b>: the backend could not say.")
VIDEO_LENGTHS = (("The whole video", 0.0), ("5 seconds", 5.0), ("10 seconds", 10.0), ("20 seconds", 20.0),
                 ("30 seconds", 30.0), ("1 minute", 60.0), ("2 minutes", 120.0), ("5 minutes", 300.0))
VIDEO_RATES = (("The video's own, up to 24 (recommended)", 0.0), ("12 per second", 12.0), ("15 per second", 15.0),
               ("24 per second", 24.0), ("30 per second", 30.0))


def h1(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h1")
    return lab


def note(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("note")
    lab.setTextFormat(Qt.RichText)
    lab.setOpenExternalLinks(True)
    return lab


def open_path(p: str | Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


VR_HELP = ("<h3>View in VR</h3>"
           "<p>The splat (or splat video) opens in the 2D2VR player in <b>Google Chrome</b> or <b>Microsoft Edge</b> "
           "on this PC. Nothing is uploaded: the page is served only to this computer (127.0.0.1).</p>"
           "<ol><li>Connect your headset to this PC: <b>Quest Link</b> (cable), <b>Air Link</b>, "
           "<b>Virtual Desktop</b> or <b>SteamVR</b>.</li>"
           "<li>In the page that opens, press <b>ENTER VR</b>.</li>"
           "<li>You start where the camera stood. Move your head to look around in 3D.</li></ol>"
           "<p><b>Controllers</b>: hold <b>A</b> or <b>X</b> to make everything smaller, <b>B</b> or <b>Y</b> to make "
           "it bigger (if people do not look life-size); click a stick to go back to the start at the original "
           "size; the <b>trigger</b> plays and pauses a splat video. Left stick moves, right stick turns.</p>"
           "<p>Without a headset the same page works with the mouse (drag, right-drag, wheel; + and − change the "
           "size).</p>")


def open_in_vr(parent, path: Path, scene=None, size: float | None = None) -> None:
    """Serve the scene (.ply / .splat) or splat video (.splv) to the bundled player and open it in a
    WebXR-capable browser."""
    from ..scene import load_scene
    from ..vr_server import get_server, view_params

    win = parent.window()
    settings = getattr(win, "settings", None)
    if settings is not None and not getattr(settings, "vr_help_seen", False):
        box = QMessageBox(parent)
        box.setWindowTitle("View in VR")
        box.setTextFormat(Qt.RichText)
        box.setText(VR_HELP)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if box.exec() != QMessageBox.Ok:
            return
        settings.vr_help_seen = True
        settings.save()
    try:
        from ..splv import frame_scene, is_splv, open_reader

        path = Path(path)
        if scene is None:
            scene = frame_scene(open_reader(path), 0) if is_splv(path) else load_scene(path)
        depth, xf = view_params(scene)
        url = get_server().share(path, depth, path.parent.parent.name, xf, size=size)
    except Exception as e:  # noqa: BLE001 - surfaced to the user
        QMessageBox.warning(parent, APP_NAME, f"Cannot open the VR player: {e}")
        return
    from ..vr_server import open_in_browser

    used = open_in_browser(url)
    if used == "default":
        box = QMessageBox(parent)
        box.setWindowTitle("View in VR")
        box.setTextFormat(Qt.RichText)
        box.setText("Chrome or Edge was not found, so the player opened in your default browser. "
                    "VR (ENTER VR) needs <b>Google Chrome</b> or <b>Microsoft Edge</b>: if the button is missing, "
                    f"paste this address into Chrome or Edge:<br><br><code>{url}</code>")
        copy = box.addButton("Copy address", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is copy:
            QGuiApplication.clipboard().setText(url)


# ============================================================ Create
class DropZone(QLabel):
    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(170)
        self.setWordWrap(True)
        self.setObjectName("dropzone")
        self.reset()

    def reset(self):
        self.setText("<div style='font-size:18px'><b>Drop photos or videos here</b></div>"
                     "<div>JPG · PNG · WebP · MP4 · MOV · MKV …</div><div style='margin-top:6px'>or click to browse</div>")

    def mousePressEvent(self, ev):  # noqa: N802
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS | VIDEO_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose photos or videos", "", f"Media ({exts})")
        if paths:
            self.filesDropped.emit(paths)

    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):  # noqa: N802
        self.filesDropped.emit([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])


class CreatePage(QWidget):
    openComponents = Signal()

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.files: list[str] = []
        v = QVBoxLayout(self)
        v.addWidget(h1("Create 3D splats and splat videos"))
        self.banner = note("")
        self.banner.setObjectName("banner")
        self.banner.hide()
        brow = QHBoxLayout()
        brow.addWidget(self.banner, 1)
        self.banner_btn = QPushButton("Open Components")
        self.banner_btn.clicked.connect(self.openComponents.emit)
        self.banner_btn.hide()
        brow.addWidget(self.banner_btn)
        v.addLayout(brow)

        top = QHBoxLayout()
        left = QVBoxLayout()
        self.drop = DropZone()
        self.drop.filesDropped.connect(self.add_files)
        left.addWidget(self.drop)
        self.selected = QLabel("No input selected.")
        self.selected.setWordWrap(True)
        left.addWidget(self.selected)
        left.addWidget(note("<b>Real 3D</b>: Apple SHARP's 3D of each photo, aligned to its measured depth "
                            "(MoGe-3) at the real lens, people with their real 3D shape (Sapiens2) and the "
                            "real-world size set from their eyes. A <b>video</b> becomes a <b>splat video</b>: every "
                            "frame made the same way, with the sound, to play in the 3D viewer and in VR."))
        top.addLayout(left, 3)

        opts = QVBoxLayout()
        box = QGroupBox("Real 3D")
        f = QFormLayout(box)
        self.quality = QComboBox()
        for m in ("auto", "quality", "fast"):
            self.quality.addItem(f"{m.capitalize()} — {MODE_HELP[m]}", m)
        self.quality.setCurrentIndex(max(self.quality.findData(win.settings.mode), 0))
        f.addRow("Quality", self.quality)
        self.ipd = QDoubleSpinBox()
        self.ipd.setRange(45.0, 80.0)
        self.ipd.setDecimals(1)
        self.ipd.setSingleStep(1.0)
        self.ipd.setSuffix(" mm")
        self.ipd.setValue(float(win.settings.ipd_mm))
        self.ipd.setToolTip("The real-world size of the scene is set from the eyes of the people in it: they are "
                            "scaled to be this far apart (adults 56-70 mm, average 63). If people look too big in VR, "
                            "lower it; too small, raise it. In VR you can also resize with the A/B buttons.")
        f.addRow("Eye distance (IPD)", self.ipd)
        self.real3d_fill = QCheckBox("Photos: AI paints what the photo hides")
        self.real3d_fill.setChecked(win.settings.real3d_fill)
        self.real3d_fill.setToolTip("The photo's splat is looked at from a little to each side; what the photo never "
                                    "saw (behind edges) is painted by Qwen-Image-Edit and added at a depth tied to the "
                                    "photo's. Shown as 'generative'. Not used for videos (too slow per frame).")
        f.addRow("", self.real3d_fill)
        self.photo_detail = QCheckBox("Photos: maximum detail (slower, more splats)")
        self.photo_detail.setChecked(win.settings.photo_detail)
        self.photo_detail.setToolTip("More of the photo's small geometry: the photo at up to 3072 px, fine relief "
                                     "from the depth model on parts of it, SHARP on parts of it, and everything the "
                                     "photo sees placed at that depth. About 2-3x the detail up close, a minute "
                                     "more, and several million splats (heavier in VR). Always on in Quality.")
        self.photo_detail.setEnabled(win.settings.mode != "quality")
        f.addRow("", self.photo_detail)
        opts.addWidget(box)

        vbox = QGroupBox("Videos → splat video")
        vf = QFormLayout(vbox)
        self.video_start = QDoubleSpinBox()
        self.video_start.setRange(0.0, 36000.0)
        self.video_start.setDecimals(1)
        self.video_start.setSuffix(" s")
        self.video_start.setValue(float(win.settings.video_start_s))
        vf.addRow("Start at", self.video_start)
        self.video_len = QComboBox()
        for label, val in VIDEO_LENGTHS:
            self.video_len.addItem(label, val)
        self.video_len.setCurrentIndex(max(self.video_len.findData(win.settings.video_seconds), 0))
        vf.addRow("Length", self.video_len)
        self.video_fps = QComboBox()
        for label, val in VIDEO_RATES:
            self.video_fps.addItem(label, val)
        self.video_fps.setCurrentIndex(max(self.video_fps.findData(win.settings.video_fps), 0))
        self.video_fps.setToolTip("Frames per second made in 3D: more is smoother motion, but slower and a larger "
                                  "file.")
        vf.addRow("Frames per second", self.video_fps)
        vf.addRow(note("Slow but steady: about 3-5 s per frame on an RTX 4080 (a minute of video at 24 fps takes "
                       "1.5-2 hours) and 2-5 GB of disk per minute. Memory stays the same however long the video "
                       "is; if the GPU runs out of memory, the job continues from the last finished frame."))
        opts.addWidget(vbox)
        opts.addStretch(1)
        for w in (self.quality, self.video_len, self.video_fps):
            w.currentIndexChanged.connect(self._save_opts)
        for w in (self.ipd, self.video_start):
            w.valueChanged.connect(self._save_opts)
        self.real3d_fill.toggled.connect(self._save_opts)
        self.photo_detail.toggled.connect(self._save_opts)
        top.addLayout(opts, 2)
        v.addLayout(top)

        brow2 = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.setMinimumHeight(44)
        self.generate_btn.clicked.connect(self.generate)
        brow2.addWidget(self.generate_btn, 3)
        self.cancel_btn = QPushButton("Cancel selected")
        self.cancel_btn.clicked.connect(self.cancel_selected)
        brow2.addWidget(self.cancel_btn, 1)
        self.cancel_all_btn = QPushButton("Cancel all")
        self.cancel_all_btn.clicked.connect(lambda: self.win.queue.cancel_all())
        brow2.addWidget(self.cancel_all_btn, 1)
        v.addLayout(brow2)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Input", "Status", "Progress", "Current step"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(120)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setPlaceholderText("Progress log")
        split.addWidget(self.log)
        split.setSizes([160, 160])
        v.addWidget(split, 1)
        self.bars: dict[str, QProgressBar] = {}
        self.refresh_backends()

    # -------------------------------------------------------------- state
    def refresh_backends(self):
        self.update_banner()

    def update_banner(self):
        from ..backends import get_backend

        msgs = []
        hw = self.win.hw
        if hw is not None and not hw.gpus:
            msgs.append("No NVIDIA GPU detected." + (" CPU mode is on (slow)." if self.win.settings.allow_cpu else
                                                     " Enable CPU mode in Settings to run slowly, or view existing "
                                                     "3D files in the Viewer."))
        av = get_backend("photo_locked").availability(self.win.ctx, None, {"mode": self.mode()})
        if not av.ok:
            msgs.append("Real 3D needs its engine and models — press 'Install recommended' on the Components page. "
                        "(" + "; ".join(av.reasons) + ")")
        self.banner.setText(" ".join(msgs))
        self.banner.setVisible(bool(msgs))
        self.banner_btn.setVisible(bool(msgs))

    def mode(self) -> str:
        return self.quality.currentData() or "auto"

    def _save_opts(self, *a):
        s = self.win.settings
        s.mode = self.mode()
        s.real3d_fill = self.real3d_fill.isChecked()
        s.photo_detail = self.photo_detail.isChecked()
        self.photo_detail.setEnabled(self.mode() != "quality")      # Quality always uses maximum detail
        s.ipd_mm = float(self.ipd.value())
        s.video_start_s = float(self.video_start.value())
        s.video_seconds = float(self.video_len.currentData())
        s.video_fps = float(self.video_fps.currentData())
        s.save()

    def add_files(self, paths: list[str]):
        ok, bad = [], []
        for p in paths:
            ext = Path(p).suffix.lower()
            if ext in (".ply", ".splat", ".splv"):
                self.win.open_scene(Path(p))
                continue
            (ok if ext in IMAGE_EXTS | VIDEO_EXTS else bad).append(p)
        if bad:
            QMessageBox.warning(self, APP_NAME, "Unsupported file type:\n" + "\n".join(Path(b).name for b in bad))
        if ok:
            self.files = ok
            names = ", ".join(Path(p).name for p in ok[:4]) + (f" and {len(ok) - 4} more" if len(ok) > 4 else "")
            hint = " — each one becomes its own result" if len(ok) > 1 else ""
            self.selected.setText(f"Selected: {names}{hint}")
            first = ok[0]
            if Path(first).suffix.lower() in IMAGE_EXTS:
                pm = QPixmap(first)
                if not pm.isNull():
                    self.drop.setPixmap(pm.scaled(420, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.drop.setText(f"<b>{Path(first).name}</b><br>video")

    def generate(self):
        if not self.files:
            QMessageBox.information(self, APP_NAME, "Drop or choose a photo or video first.")
            return
        from copy import deepcopy

        opts_base = self.win.settings.job_options()
        for p in self.files:
            self.win.queue.add(Path(p), deepcopy(opts_base))
        self.files = []
        self.selected.setText("Added to the queue.")
        self.drop.reset()

    def cancel_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            self.win.queue.cancel(self.table.item(r, 0).data(Qt.UserRole))

    def refresh_queue(self):
        jobs = self.win.queue.jobs
        self.table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            it = QTableWidgetItem(j.input.name)
            it.setData(Qt.UserRole, j.id)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(j.state))
            bar = self.bars.get(j.id)
            if bar is None:
                bar = QProgressBar()
                bar.setRange(0, 1000)
                self.bars[j.id] = bar
            bar.setValue(int(j.progress * 1000) if j.state != "succeeded" else 1000)
            self.table.setCellWidget(r, 2, bar)
            msg = j.message if j.state == "running" else ((j.report or {}).get("error") or {}).get("message", "")
            self.table.setItem(r, 3, QTableWidgetItem((msg or "")[:300]))

    def on_event(self, ev: dict):
        k = ev.get("event")
        if k == "progress":
            self.log.appendPlainText(f"[{ev.get('value', 0) * 100:5.1f}%] {ev.get('message', '')}")
        elif k == "log":
            self.log.appendPlainText("    " + str(ev.get("message", "")))
        elif k == "finished":
            err = ev.get("error") or {}
            self.log.appendPlainText(f"== {ev.get('status')} {err.get('message', '')}")
        self.refresh_queue()


# ============================================================ Results
class ResultsPage(QWidget):
    exploreScene = Signal(str)

    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(h1("Results"))
        head.addStretch(1)
        rb = QPushButton("Refresh")
        rb.clicked.connect(self.refresh)
        head.addWidget(rb)
        v.addLayout(head)
        split = QSplitter()
        self.list = QListWidget()
        self.list.setIconSize(QSize(128, 64))
        self.list.currentItemChanged.connect(self.show_item)
        split.addWidget(self.list)
        right = QWidget()
        rv = QVBoxLayout(right)
        self.preview = QLabel("Select a result.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        rv.addWidget(self.preview, 3)
        grid = QGridLayout()
        self.btn = {}
        for i, (key, label) in enumerate([("explore", "Play / explore in 3D"), ("vr", "View in VR"),
                                          ("folder", "Open folder"), ("export", "Copy results to…"),
                                          ("delete", "Delete")]):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=key: self.action(k))
            grid.addWidget(b, i // 5, i % 5)
            self.btn[key] = b
        rv.addLayout(grid)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        rv.addWidget(self.details, 2)
        rv.addWidget(note("<b>View in VR</b> plays the splat or splat video in a PC-connected headset (Quest Link / "
                          "Air Link / Virtual Desktop / SteamVR). A/X and B/Y resize it if people do not look "
                          "life-size. Each result folder has a README.txt explaining every file."))
        split.addWidget(right)
        split.setSizes([380, 700])
        v.addWidget(split, 1)
        self.reports: dict[str, dict] = {}
        self.refresh()

    def refresh(self):
        from ..paths import app_paths

        cur = self.list.currentItem().data(Qt.UserRole) if self.list.currentItem() else None
        self.list.clear()
        self.reports.clear()
        root = app_paths().jobs
        dirs = sorted([d for d in root.glob("*") if (d / "run_report.json").exists()], reverse=True) \
            if root.exists() else []
        for d in dirs[:500]:
            try:
                rep = json.loads((d / "run_report.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            self.reports[str(d)] = self.relocate(d, rep)
            name = Path((rep.get("input") or {}).get("path", d.name)).name
            status = rep.get("status", "?")
            backend = (rep.get("backend") or {}).get("name") or "—"
            it = QListWidgetItem(f"{name}\n{d.name[:15]} · {status} · {backend}")
            it.setData(Qt.UserRole, str(d))
            thumb = self._thumb(d, rep)
            if thumb:
                it.setIcon(QIcon(thumb))
            self.list.addItem(it)
            if str(d) == cur:
                self.list.setCurrentItem(it)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        if not self.list.count():
            self.preview.setText("No results yet — generate something on the Create page.")

    def _thumb(self, d: Path, rep: dict) -> str | None:
        for name in ("views_sheet.png", "thumb.jpg"):
            if (d / "export" / name).exists():
                return str(d / "export" / name)
        frames = sorted((d / "frames").glob("frame_*.*"))
        return str(frames[0]) if frames else None

    def _current(self) -> tuple[Path | None, dict]:
        it = self.list.currentItem()
        if not it:
            return None, {}
        d = it.data(Qt.UserRole)
        return Path(d), self.reports.get(d, {})

    @staticmethod
    def relocate(d: Path, rep: dict) -> dict:
        """Reports keep absolute paths: after the data folder moved (1.x 2D2VR180 → 2D2VR) point them into
        this job folder again."""
        def fix(v):
            if isinstance(v, str) and d.name in Path(v).parts and not Path(v).exists():
                parts = Path(v).parts
                return str(d.joinpath(*parts[parts.index(d.name) + 1:]))
            return v

        out = rep.get("outputs")
        if isinstance(out, dict):
            rep["outputs"] = {k: fix(v) for k, v in out.items()}
        return rep

    def show_item(self, *a):
        d, rep = self._current()
        if d is None:
            return
        out = rep.get("outputs") or {}
        t = self._thumb(d, rep)
        if t:
            pm = QPixmap(t)
            self.preview.setPixmap(pm.scaled(self.preview.width() or 700, self.preview.height() or 300,
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.setText("(no preview)")
        self.btn["explore"].setEnabled(bool(out.get("scene_ply")))
        self.btn["vr"].setEnabled(bool(out.get("scene_ply")))
        html = [f"<h3>{Path((rep.get('input') or {}).get('path', '')).name}</h3>",
                f"<p>Status: <b>{rep.get('status')}</b> · backend: {(rep.get('backend') or {}).get('name', '—')}"
                f" · time: {rep.get('runtime_s', '?')} s</p>"]
        if rep.get("error"):
            html.append(f"<p style='color:#e66'><b>Error</b> [{rep['error'].get('code')}]: "
                        f"{rep['error'].get('message')}</p>")
        fs = ((rep.get("backend") or {}).get("extra") or {}).get("fill_status")
        if fs:
            html.append(f"<p><b>AI fill of what the photo hides</b>: {fs}</p>")
        if out.get("splv"):
            html.append(f"<p><b>Splat video</b>: {out.get('frames')} frames at {out.get('fps')} fps, "
                        f"{out.get('splats_per_frame') or 0:,} splats per frame"
                        + (", with sound" if out.get("audio") else ", no sound") + "</p>")
        cov = (rep.get("coverage") or {}).get("by_splat")
        if cov:
            html.append("<p><b>Geometry provenance</b>: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items() if v)
                        + f"<br><small>{PROVENANCE_HELP}</small></p>")
        for w in rep.get("warnings", []):
            html.append(f"<p>⚠ {w}</p>")
        files = [out.get(k) for k in ("splv", "scene_ply", "splat", "obj") if out.get(k)]
        if files:
            html.append("<p><b>Files</b><br>" + "<br>".join(Path(f).name for f in files) + "</p>")
        self.details.setHtml("".join(html))

    def action(self, key: str):
        d, rep = self._current()
        if d is None:
            return
        out = rep.get("outputs") or {}
        if key == "explore" and out.get("scene_ply"):
            self.exploreScene.emit(out.get("splv") or out["scene_ply"])
        elif key == "vr" and out.get("scene_ply"):
            open_in_vr(self, Path(out.get("splv") or out["scene_ply"]))
        elif key == "folder":
            open_path(d / "export" if (d / "export").exists() else d)
        elif key == "export":
            dst = QFileDialog.getExistingDirectory(self, "Copy results to")
            if dst and (d / "export").exists():
                target = Path(dst) / f"{Path((rep.get('input') or {}).get('path', 'result')).stem}_{d.name}"
                shutil.copytree(d / "export", target, dirs_exist_ok=True)
                shutil.copy2(d / "run_report.json", target / "run_report.json")
                open_path(target)
        elif key == "delete":
            if QMessageBox.question(self, APP_NAME, f"Delete this result ({d.name}) and its files?") == QMessageBox.Yes:
                shutil.rmtree(d, ignore_errors=True)
                self.refresh()


# ============================================================ Components
class ComponentCard(QFrame):
    def __init__(self, page, comp):
        super().__init__()
        self.page = page
        self.comp = comp
        self.setObjectName("card")
        g = QGridLayout(self)
        title = QLabel(f"<b>{comp.name}</b>" + ("  <span style='color:#6c6'>recommended</span>" if comp.recommended else ""))
        g.addWidget(title, 0, 0, 1, 3)
        g.addWidget(note(comp.feature + (f"<br><i>{comp.note}</i>" if comp.note else "")), 1, 0, 1, 3)
        self.info = note("")
        g.addWidget(self.info, 2, 0, 1, 3)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.hide()
        g.addWidget(self.bar, 3, 0, 1, 2)
        self.msg = QLabel("")
        self.msg.setObjectName("note")
        g.addWidget(self.msg, 4, 0, 1, 3)
        self.install = QPushButton("Install")
        self.install.clicked.connect(lambda: page.install([comp.id]))
        g.addWidget(self.install, 0, 3)
        self.remove = QPushButton("Remove")
        self.remove.clicked.connect(lambda: page.remove(comp))
        g.addWidget(self.remove, 1, 3)
        if comp.kind == "model":
            self.verify = QPushButton("Verify")
            self.verify.clicked.connect(lambda: page.verify(comp))
            g.addWidget(self.verify, 2, 3)

    def refresh(self):
        from ..components import component_status

        st = component_status(self.page.win.ctx, self.comp)
        size = f"{st['size_gb']:.1f} GB" if st["size_gb"] else "size shown after first download"
        lic = st["license"]
        if st["license_url"]:
            lic = f"<a href='{st['license_url']}'>{lic}</a>"
        com = "" if st["commercial_use"] is None else (" · commercial use: " + ("yes" if st["commercial_use"] else "<b>no</b>"))
        state = "<b style='color:#6c6'>Installed</b>" if st["installed"] else "Not installed"
        self.info.setText(f"{state} · {size} · licence: {lic}{com}")
        busy = self.page.win.installer.busy()
        self.install.setEnabled(not st["installed"] and not busy)
        self.install.setText("Installed" if st["installed"] else "Install")
        self.remove.setEnabled(st["installed"] and not busy)
        if hasattr(self, "verify"):
            self.verify.setEnabled(st["installed"])


class ComponentsPage(QWidget):
    changed = Signal()

    def __init__(self, win):
        super().__init__()
        self.win = win
        from ..components import COMPONENTS
        from ..paths import app_paths

        v = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(h1("Components"))
        head.addStretch(1)
        self.rec_btn = QPushButton("Install recommended")
        self.rec_btn.setObjectName("primary")
        self.rec_btn.clicked.connect(self.install_recommended)
        head.addWidget(self.rec_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(lambda: self.win.installer.cancel())
        head.addWidget(self.stop_btn)
        v.addLayout(head)
        v.addWidget(note(f"Engines and AI models are downloaded once from their original publishers into "
                         f"<code>{app_paths().root}</code>. Nothing you process ever leaves this computer. "
                         "Downloads resume if interrupted and are verified before use."))
        orow = QHBoxLayout()
        self.old_note = note("")
        orow.addWidget(self.old_note, 1)
        self.old_btn = QPushButton("Delete them")
        self.old_btn.clicked.connect(self.delete_unused)
        orow.addWidget(self.old_btn)
        v.addLayout(orow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        self.cards = {}
        for c in COMPONENTS:
            card = ComponentCard(self, c)
            self.cards[c.id] = card
            iv.addWidget(card)
        iv.addStretch(1)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        win.installer.progress.connect(self.on_progress)
        win.installer.finished.connect(self.on_finished)
        win.installer.idle.connect(self.refresh)
        self.refresh()

    def refresh(self):
        for c in self.cards.values():
            c.refresh()
        ctx = self.win.ctx
        try:
            models, runtimes = ctx.models.unused(), ctx.runtimes.unused()
        except (OSError, AttributeError):
            models, runtimes = [], []
        if models or runtimes:
            gb = sum(b for _, b in models) / 2**30
            names = [n for n, _ in models] + runtimes
            self.old_note.setText(f"<b>Not used any more</b> (downloaded by earlier versions): {', '.join(names)}"
                                  + (f" — models {gb:.1f} GB" if models else "")
                                  + (" plus the engines' folders" if runtimes else "") + ".")
        self.old_note.setVisible(bool(models or runtimes))
        self.old_btn.setVisible(bool(models or runtimes))
        self.old_btn.setEnabled(not self.win.installer.busy())
        self.rec_btn.setEnabled(not self.win.installer.busy())
        self.stop_btn.setEnabled(self.win.installer.busy())
        self.changed.emit()

    def delete_unused(self):
        if QMessageBox.question(self, APP_NAME, "Delete the models and engines that this version no longer "
                                                "uses? They can't be used by 2D2VR any more.") != QMessageBox.Yes:
            return
        freed = self.win.ctx.models.delete_unused()
        self.win.ctx.runtimes.delete_unused()
        QMessageBox.information(self, APP_NAME, f"Deleted (models: {freed / 2**30:.1f} GB freed, plus the old "
                                                f"engines).")
        self.refresh()

    def install_recommended(self):
        from ..components import COMPONENTS, component_status

        todo = [c.id for c in COMPONENTS if c.recommended and not component_status(self.win.ctx, c)["installed"]]
        if not todo:
            QMessageBox.information(self, APP_NAME, "All recommended components are installed.")
            return
        self.install(todo)

    def install(self, ids: list[str]):
        from ..components import component_status, install_order

        comps = [c for c in install_order(ids) if not component_status(self.win.ctx, c)["installed"]]
        if not comps:
            return
        mm = self.win.ctx.models
        gated = [c for c in comps if c.kind == "model" and mm.entries[c.target].gated and not mm.is_installed(c.target)]
        if gated and not self.win.settings.hf_token:
            QMessageBox.information(
                self, APP_NAME,
                "<p>" + ", ".join(c.name for c in gated) + " is published behind a free licence agreement.</p>"
                "<ol><li>Sign in at <a href='https://huggingface.co'>huggingface.co</a> and accept the licence on "
                + " / ".join(f"<a href='{mm.entries[c.target].source}'>{mm.entries[c.target].source}</a>"
                             for c in gated) + "</li><li>Create an access token (Settings → Access Tokens, type "
                "'Read') and paste it in 2D2VR → Settings → Hugging Face token.</li>"
                "<li>Install it again.</li></ol><p>The other components will be installed now.</p>")
            comps = [c for c in comps if c not in gated]
            if not comps:
                return
        if not self._accept_licences(comps):
            return
        total = sum((component_status(self.win.ctx, c)["size_gb"] or 0) for c in comps)
        free = shutil.disk_usage(self.win.ctx.models.root).free / 2**30
        if total and free < total * 1.2:
            QMessageBox.warning(self, APP_NAME, f"Not enough free disk space: about {total:.1f} GB needed, "
                                                f"{free:.1f} GB free.")
            return
        self.win.installer.enqueue(comps)
        self.refresh()

    def _accept_licences(self, comps) -> bool:
        mm = self.win.ctx.models
        models = [c for c in comps if c.kind == "model" and not mm.license_accepted(c.target)]
        if not models:
            return True
        rows = []
        for c in models:
            e = mm.entries[c.target]
            rows.append(f"<p><b>{e.display_name}</b><br>Source: {e.source}<br>Licence: "
                        f"<a href='{e.license_url}'>{e.license}</a><br>Commercial use: "
                        f"{'yes' if e.commercial_use else '<b>NO</b>'} · Redistributable: "
                        f"{'yes' if e.redistributable else 'no'}<br><small>{e.notes}</small></p>")
        box = QMessageBox(self)
        box.setWindowTitle("Model licences")
        box.setTextFormat(Qt.RichText)
        box.setText("These models are downloaded from their original publishers. Please read their licences:"
                    + "".join(rows) + "<p>Do you accept these licences?</p>")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            return False
        for c in models:
            mm.accept_license(c.target)
        return True

    def remove(self, comp):
        from ..components import remove_component

        if QMessageBox.question(self, APP_NAME, f"Remove {comp.name}?") == QMessageBox.Yes:
            remove_component(self.win.ctx, comp)
            self.refresh()

    def verify(self, comp):
        probs = self.win.ctx.models.verify(comp.target)
        QMessageBox.information(self, APP_NAME, "Verified: file hashes match." if not probs else "\n".join(probs))

    def on_progress(self, cid: str, frac: float, msg: str):
        card = self.cards.get(cid)
        if not card:
            return
        card.bar.show()
        if frac < 0:
            card.bar.setRange(0, 0)
        else:
            card.bar.setRange(0, 1000)
            card.bar.setValue(int(frac * 1000))
        card.msg.setText(msg[-160:])
        self.stop_btn.setEnabled(True)
        self.rec_btn.setEnabled(False)

    def on_finished(self, cid: str, ok: bool, msg: str):
        card = self.cards.get(cid)
        if card:
            card.bar.setRange(0, 1000)
            card.bar.setValue(1000 if ok else 0)
            card.bar.setVisible(not ok)
            card.msg.setText("Installed." if ok else f"Failed: {msg[-300:]}")
        if not ok and msg != "cancelled":
            QMessageBox.warning(self, APP_NAME, f"Installing '{cid}' failed:\n\n{msg[-800:]}")
        self.refresh()


# ============================================================ Settings
class SettingsPage(QWidget):
    changed = Signal()

    def __init__(self, win):
        super().__init__()
        self.win = win
        s = win.settings
        v = QVBoxLayout(self)
        v.addWidget(h1("Settings"))
        form = QFormLayout()
        self.profile = QComboBox()
        self.profile.addItem("Personal / research use (all models)", "personal_research")
        self.profile.addItem("Commercial use (only commercially licensed models)", "commercial")
        self.profile.setCurrentIndex(0 if s.license_profile == "personal_research" else 1)
        form.addRow("Licence profile", self.profile)
        self.cpu = QCheckBox("Allow CPU inference when no NVIDIA GPU is available (much slower)")
        self.cpu.setChecked(s.allow_cpu)
        form.addRow("CPU mode", self.cpu)
        orow = QHBoxLayout()
        self.outdir = QLineEdit(s.output_dir)
        self.outdir.setPlaceholderText("(keep results in the app's jobs folder only)")
        ob = QPushButton("Browse…")
        ob.clicked.connect(self._browse)
        orow.addWidget(self.outdir, 1)
        orow.addWidget(ob)
        form.addRow("Also copy results to", orow)
        self.token = QLineEdit(s.hf_token)
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("only for gated models; stored on this computer only")
        form.addRow("Hugging Face token", self.token)
        v.addLayout(form)
        from ..paths import app_paths

        v.addWidget(note(f"Data folder: <code>{app_paths().root}</code>. To use another drive, set the "
                         "environment variable <code>TWOD2VR_HOME</code> before starting the app."))
        v.addStretch(1)
        self.profile.currentIndexChanged.connect(self.save)
        self.cpu.toggled.connect(self.save)
        for w in (self.outdir, self.token):
            w.editingFinished.connect(self.save)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Copy results to")
        if d:
            self.outdir.setText(d)
            self.save()

    def save(self, *a):
        s = self.win.settings
        s.license_profile = self.profile.currentData()
        s.allow_cpu = self.cpu.isChecked()
        s.output_dir = self.outdir.text().strip()
        s.hf_token = self.token.text().strip()
        s.sanitize()
        s.save()
        self.win.ctx.license_profile = s.license_profile
        self.win.ctx.allow_cpu = s.allow_cpu
        self.changed.emit()


# ============================================================ Diagnostics
class DiagnosticsPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(h1("Diagnostics"))
        head.addStretch(1)
        for label, fn in (("Refresh", self.refresh_hw), ("Copy to clipboard", self.copy),
                          ("Open data folder", self.open_data)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            head.addWidget(b)
        v.addLayout(head)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        v.addWidget(self.text, 1)
        self.last_report: dict | None = None
        self.render()

    def refresh_hw(self):
        self.win.refresh_hardware()

    def open_data(self):
        from ..paths import app_paths

        open_path(app_paths().root)

    def copy(self):
        QGuiApplication.clipboard().setText(self.text.toPlainText())

    def render(self):
        from ..backends import all_backends
        from ..hardware import format_report

        html = [f"<h3>{APP_NAME} {__version__}</h3>"]
        if self.win.hw:
            html.append("<h4>Hardware</h4><pre>" + format_report(self.win.hw) + "</pre>")
        html.append("<h4>Backends</h4><table border=1 cellspacing=0 cellpadding=4>"
                    "<tr><th>Backend</th><th>Maturity</th><th>Commercial</th><th>Available</th><th>Why not</th></tr>")
        for b in all_backends():
            av = b.availability(self.win.ctx, self.win.hw, {"mode": self.win.settings.mode})
            html.append(f"<tr><td>{b.display_name}</td><td>{b.maturity}</td><td>{'yes' if b.commercial_use else 'no'}"
                        f"</td><td>{'yes' if av.ok else 'no'}</td><td>{'<br>'.join(av.reasons)}</td></tr>")
        html.append("</table>")
        if self.last_report:
            html.append("<h4>Last job report</h4><pre>" +
                        json.dumps({k: self.last_report.get(k) for k in ("status", "error", "selection", "backend",
                                                                         "coverage", "validation", "timings_s",
                                                                         "warnings")}, indent=2, default=str)
                        + "</pre>")
        self.text.setHtml("".join(html))


# ============================================================ Viewer
class ViewerPage(QWidget):
    def __init__(self, win):
        super().__init__()
        from .viewer import SceneViewer

        self.win = win
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(h1("3D viewer"))
        head.addStretch(1)
        ob = QPushButton("Open .ply / .splat / .splv…")
        ob.clicked.connect(self.open_file)
        head.addWidget(ob)
        head.addWidget(QLabel("Size in VR"))
        self.vr_size = QSpinBox()
        self.vr_size.setRange(20, 500)
        self.vr_size.setSingleStep(5)
        self.vr_size.setSuffix(" %")
        self.vr_size.setValue(100)
        self.vr_size.setToolTip("The size the scene starts with in VR (100 % = as made). Use it when the people do "
                                "not look life-size; in VR, A/X and B/Y change it too.")
        head.addWidget(self.vr_size)
        self.xr_btn = QPushButton("View in VR")
        self.xr_btn.clicked.connect(self.view_in_vr)
        self.xr_btn.setEnabled(False)
        head.addWidget(self.xr_btn)
        v.addLayout(head)
        self.viewer = SceneViewer()
        v.addWidget(self.viewer, 1)
        self.path: Path | None = None

    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open a 3D scene or splat video", "",
                                           "Gaussian splats (*.ply *.splat *.splv)")
        if p:
            self.win.open_scene(Path(p))

    def load(self, path: Path) -> bool:
        from ..scene import SceneError, load_scene
        from ..splv import is_splv

        try:
            if is_splv(path):
                self.viewer.set_video(path)
            else:
                self.viewer.set_scene(load_scene(path))
        except (SceneError, OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, APP_NAME, f"Cannot open {path.name}: {e}")
            return False
        self.path = path
        self.xr_btn.setEnabled(True)
        return True

    def view_in_vr(self):
        if not self.path:
            return
        self.viewer.set_playing(False)
        from ..splv import is_splv

        open_in_vr(self, self.path, None if is_splv(self.path) else self.viewer.scene,
                   size=self.vr_size.value() / 100.0)
