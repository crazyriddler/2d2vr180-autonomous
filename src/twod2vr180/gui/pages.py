"""Application pages: Create, Results, 3D Viewer, Components, Settings, Diagnostics."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..media import IMAGE_EXTS, VIDEO_EXTS

MODE_HELP = {
    "auto": "Best available backend for the input (recommended).",
    "quality": "Largest models and highest resolution. Slower.",
    "fast": "Small models, quick preview-quality results.",
}
PROVENANCE_HELP = ("<b>Observed</b>: reconstructed from several views. <b>Inferred</b>: seen in the input but its "
                   "depth is predicted by a network. <b>Interpolated</b>: small gaps filled from neighbouring "
                   "background. <b>Unknown</b>: never seen — shown black, never invented.")


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


VR_HELP = ("<h3>View the 3D splat in VR</h3>"
           "<p>The scene opens in a WebXR viewer in <b>Google Chrome</b> or <b>Microsoft Edge</b> on this PC. "
           "Nothing is uploaded: the page is served only to this computer (127.0.0.1).</p>"
           "<ol><li>Connect your headset to this PC: <b>Quest Link</b> (cable), <b>Air Link</b>, "
           "<b>Virtual Desktop</b> or <b>SteamVR</b>.</li>"
           "<li>In the page that opens, press <b>ENTER VR</b> at the bottom.</li>"
           "<li>You start where the original camera stood. Move your head to look around the 3D scene.</li></ol>"
           "<p>Without a headset the same page works with the mouse (drag, right-drag, wheel).</p>"
           "<p><i>A single photo only contains what the camera saw: step sideways and you will see empty space "
           "behind objects.</i> For watching on a standalone headset, use the VR180 video instead "
           "(Results → Play VR180 video; copy the _180_LR.mp4 to the headset).</p>")


def open_in_vr(parent, ply: Path, scene=None) -> None:
    """Serve the scene to the bundled WebXR viewer and open it in a WebXR-capable browser."""
    from ..scene import load_scene
    from ..vr_server import get_server, scene_depth

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
        scene = scene if scene is not None else load_scene(Path(ply))
        url = get_server().share(Path(ply), scene_depth(scene), Path(ply).parent.parent.name)
    except Exception as e:  # noqa: BLE001 - surfaced to the user
        QMessageBox.warning(parent, APP_NAME, f"Cannot open the VR viewer: {e}")
        return
    from ..vr_server import open_in_browser

    used = open_in_browser(url)
    if used == "default":
        box = QMessageBox(parent)
        box.setWindowTitle("View in VR")
        box.setTextFormat(Qt.RichText)
        box.setText("Chrome or Edge was not found, so the viewer opened in your default browser. "
                    "VR (ENTER VR) needs <b>Google Chrome</b> or <b>Microsoft Edge</b>: if the button says "
                    f"'VR not supported', paste this address into Chrome or Edge:<br><br><code>{url}</code>")
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
        v.addWidget(h1("Create 3D and VR180"))
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
        top.addLayout(left, 3)

        opts = QVBoxLayout()
        qbox = QGroupBox("Quality")
        ql = QVBoxLayout(qbox)
        self.mode_group = QButtonGroup(self)
        for m in ("auto", "quality", "fast"):
            rb = QRadioButton(f"{m.capitalize()} — {MODE_HELP[m]}")
            rb.setProperty("mode", m)
            self.mode_group.addButton(rb)
            ql.addWidget(rb)
            if m == win.settings.mode:
                rb.setChecked(True)
        self.mode_group.buttonToggled.connect(self._save_opts)
        opts.addWidget(qbox)

        obox = QGroupBox("Outputs")
        of = QFormLayout(obox)
        of.addRow(QLabel("Always: 3D Gaussian splat (.ply, .splat) and, when available, a textured mesh (.obj)."))
        self.vr = QCheckBox("VR180 stereo")
        self.vr.setChecked(win.settings.vr180)
        self.sbs = QCheckBox("Side-by-side (LR)")
        self.sbs.setChecked(win.settings.layout_sbs)
        self.tb = QCheckBox("Top/bottom (TB)")
        self.tb.setChecked(win.settings.layout_tb)
        row = QHBoxLayout()
        for w in (self.vr, self.sbs, self.tb):
            row.addWidget(w)
            w.toggled.connect(self._save_opts)
        of.addRow(row)
        self.projection = QComboBox()
        self.projection.addItem("VR180 (180° half-equirectangular)", "equirect180")
        self.projection.addItem("Flat 3D (source field of view)", "flat")
        self.projection.setCurrentIndex(0 if win.settings.projection == "equirect180" else 1)
        self.projection.currentIndexChanged.connect(self._save_opts)
        of.addRow("Projection", self.projection)
        self.backend = QComboBox()
        of.addRow("Backend", self.backend)
        opts.addWidget(obox)
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
        from ..backends import all_backends

        cur = self.backend.currentData()
        self.backend.blockSignals(True)
        self.backend.clear()
        self.backend.addItem("Automatic (recommended)", None)
        for b in all_backends():
            if b.maturity == "unsupported":
                continue
            av = b.availability(self.win.ctx, self.win.hw, {"mode": self.mode()})
            self.backend.addItem(b.display_name + ("" if av.ok else "  — unavailable"), b.id)
        idx = self.backend.findData(cur)
        self.backend.setCurrentIndex(max(idx, 0))
        self.backend.blockSignals(False)
        self.update_banner()

    def update_banner(self):
        msgs = []
        hw = self.win.hw
        if hw is not None and not hw.gpus:
            msgs.append("No NVIDIA GPU detected." + (" CPU mode is on (slow)." if self.win.settings.allow_cpu else
                                                     " Enable CPU mode in Settings to run slowly, or view existing "
                                                     "3D files in the Viewer."))
        if not self.win.ctx.runtimes.is_installed("photo-cu128"):
            msgs.append("The photo engine is not installed yet — install the recommended components first.")
        elif not any(self.win.ctx.models.is_installed(m) for m in
                     ("moge-2-vitl-normal", "moge-2-vits-normal", "depth-anything-v2-small", "sharp")):
            msgs.append("No reconstruction model is installed yet.")
        self.banner.setText(" ".join(msgs))
        self.banner.setVisible(bool(msgs))
        self.banner_btn.setVisible(bool(msgs))

    def mode(self) -> str:
        b = self.mode_group.checkedButton()
        return b.property("mode") if b else "auto"

    def _save_opts(self, *a):
        s = self.win.settings
        s.mode = self.mode()
        s.vr180 = self.vr.isChecked()
        s.layout_sbs = self.sbs.isChecked()
        s.layout_tb = self.tb.isChecked()
        s.projection = self.projection.currentData()
        self.sbs.setEnabled(s.vr180)
        self.tb.setEnabled(s.vr180)
        self.projection.setEnabled(s.vr180)
        s.save()

    def add_files(self, paths: list[str]):
        ok, bad = [], []
        for p in paths:
            ext = Path(p).suffix.lower()
            if ext in (".ply", ".splat"):
                self.win.open_scene(Path(p))
                continue
            (ok if ext in IMAGE_EXTS | VIDEO_EXTS else bad).append(p)
        if bad:
            QMessageBox.warning(self, APP_NAME, "Unsupported file type:\n" + "\n".join(Path(b).name for b in bad))
        if ok:
            self.files = ok
            names = ", ".join(Path(p).name for p in ok[:4]) + (f" and {len(ok) - 4} more" if len(ok) > 4 else "")
            self.selected.setText(f"Selected: {names}")
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
        opts_base = self.win.settings.job_options(backend=self.backend.currentData())
        for p in self.files:
            from copy import deepcopy

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
        for i, (key, label) in enumerate([("explore", "Explore in 3D"), ("vr", "View in VR (6DoF)"),
                                          ("vrimg", "Open VR180 image"), ("vrvid", "Play VR180 video"),
                                          ("folder", "Open folder"), ("export", "Copy results to…"),
                                          ("delete", "Delete")]):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=key: self.action(k))
            grid.addWidget(b, i // 4, i % 4)
            self.btn[key] = b
        rv.addLayout(grid)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        rv.addWidget(self.details, 2)
        rv.addWidget(note("<b>Headset, two ways:</b> <b>View in VR</b> lets you look around the 3D splat with a "
                          "PC-connected headset (Quest Link / Air Link / SteamVR). For a standalone headset, copy the "
                          "<code>vr180\\*_180_LR.mp4</code> to it (Quest: USB → Movies) and play it as 180° "
                          "side-by-side. Each result folder has a README.txt explaining every file."))
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
            self.reports[str(d)] = rep
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
        stills = ((rep.get("outputs") or {}).get("vr180") or {}).get("stills") or []
        if stills and Path(stills[0]["image"]).exists():
            return stills[0]["image"]
        frames = sorted((d / "frames").glob("*.png"))
        return str(frames[0]) if frames else None

    def _current(self) -> tuple[Path | None, dict]:
        it = self.list.currentItem()
        if not it:
            return None, {}
        d = it.data(Qt.UserRole)
        return Path(d), self.reports.get(d, {})

    def show_item(self, *a):
        d, rep = self._current()
        if d is None:
            return
        out = rep.get("outputs") or {}
        vr = out.get("vr180") or {}
        t = self._thumb(d, rep)
        if t:
            pm = QPixmap(t)
            self.preview.setPixmap(pm.scaled(self.preview.width() or 700, self.preview.height() or 300,
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.setText("(no preview)")
        self.btn["explore"].setEnabled(bool(out.get("scene_ply")))
        self.btn["vr"].setEnabled(bool(out.get("scene_ply")))
        self.btn["vrimg"].setEnabled(bool(vr.get("stills")))
        self.btn["vrvid"].setEnabled(bool(vr.get("videos")))
        html = [f"<h3>{Path((rep.get('input') or {}).get('path', '')).name}</h3>",
                f"<p>Status: <b>{rep.get('status')}</b> · backend: {(rep.get('backend') or {}).get('name', '—')}"
                f" · time: {rep.get('runtime_s', '?')} s</p>"]
        if rep.get("error"):
            html.append(f"<p style='color:#e66'><b>Error</b> [{rep['error'].get('code')}]: "
                        f"{rep['error'].get('message')}</p>")
        cov = (rep.get("coverage") or {}).get("by_splat")
        if cov:
            html.append("<p><b>Geometry provenance</b>: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items() if v)
                        + f"<br><small>{PROVENANCE_HELP}</small></p>")
        for st in vr.get("stills", []):
            md = st["metadata"]
            c = md["coverage"]["left"]
            html.append(f"<p><b>{Path(st['image']).name}</b>: content {md['content_fov_deg']['horizontal_deg']:.0f}°×"
                        f"{md['content_fov_deg']['vertical_deg']:.0f}°, full VR180: {'yes' if md['is_full_vr180'] else 'no'}"
                        f"; left eye covered {c['covered_fraction']:.0%}, interpolated "
                        f"{c.get('hole_filled_fraction', 0) + c.get('crack_filled_fraction', 0):.1%}, unknown "
                        f"{c['unknown_fraction']:.0%}; renderer {md.get('renderer', '?')}</p>")
        for w in rep.get("warnings", []):
            html.append(f"<p>⚠ {w}</p>")
        files = [out.get(k) for k in ("scene_ply", "splat", "obj") if out.get(k)]
        files += [v["path"] for v in vr.get("videos", [])]
        if files:
            html.append("<p><b>Files</b><br>" + "<br>".join(Path(f).name for f in files) + "</p>")
        self.details.setHtml("".join(html))

    def action(self, key: str):
        d, rep = self._current()
        if d is None:
            return
        out = rep.get("outputs") or {}
        vr = out.get("vr180") or {}
        if key == "explore" and out.get("scene_ply"):
            self.exploreScene.emit(out["scene_ply"])
        elif key == "vr" and out.get("scene_ply"):
            open_in_vr(self, Path(out["scene_ply"]))
        elif key == "vrimg" and vr.get("stills"):
            open_path(vr["stills"][0]["image"])
        elif key == "vrvid" and vr.get("videos"):
            open_path(vr["videos"][0]["path"])
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
        self.rec_btn.setEnabled(not self.win.installer.busy())
        self.stop_btn.setEnabled(self.win.installer.busy())
        self.changed.emit()

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
        self.renderer = QComboBox()
        for label, val in (("Automatic (GPU when available)", "auto"), ("GPU — gsplat 3DGS", "gpu"),
                           ("CPU reference renderer", "cpu")):
            self.renderer.addItem(label, val)
        self.renderer.setCurrentIndex(self.renderer.findData(s.renderer))
        form.addRow("VR180 renderer", self.renderer)
        self.fill = QCheckBox("Fill small disocclusion gaps from the background (reported as interpolated)")
        self.fill.setChecked(s.fill_holes)
        form.addRow("Hole filling", self.fill)
        self.ipd = QDoubleSpinBox()
        self.ipd.setRange(0, 200)
        self.ipd.setSuffix(" mm")
        self.ipd.setValue(s.eye_separation_mm)
        form.addRow("Eye separation", self.ipd)
        self.eye = QSpinBox()
        self.eye.setRange(256, 4096)
        self.eye.setSingleStep(256)
        self.eye.setValue(s.eye_resolution)
        form.addRow("VR180 image resolution (per eye)", self.eye)
        self.veye = QSpinBox()
        self.veye.setRange(256, 4096)
        self.veye.setSingleStep(256)
        self.veye.setValue(s.video_eye_resolution)
        form.addRow("VR180 video resolution (per eye)", self.veye)
        self.secs = QDoubleSpinBox()
        self.secs.setRange(0.5, 120)
        self.secs.setSuffix(" s")
        self.secs.setValue(s.still_video_seconds)
        form.addRow("VR180 video length for photos", self.secs)
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
        self.token.setPlaceholderText("only needed for gated models; stored on this computer only")
        form.addRow("Hugging Face token", self.token)
        v.addLayout(form)
        from ..paths import app_paths

        v.addWidget(note(f"Data folder: <code>{app_paths().root}</code>. To use another drive, set the "
                         "environment variable <code>TWOD2VR180_HOME</code> before starting the app."))
        v.addStretch(1)
        for w in (self.profile, self.renderer):
            w.currentIndexChanged.connect(self.save)
        for w in (self.fill, self.cpu):
            w.toggled.connect(self.save)
        for w in (self.ipd, self.eye, self.veye, self.secs):
            w.valueChanged.connect(self.save)
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
        s.renderer = self.renderer.currentData()
        s.fill_holes = self.fill.isChecked()
        s.eye_separation_mm = self.ipd.value()
        s.eye_resolution = self.eye.value()
        s.video_eye_resolution = self.veye.value()
        s.still_video_seconds = self.secs.value()
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
        ob = QPushButton("Open .ply / .splat…")
        ob.clicked.connect(self.open_file)
        head.addWidget(ob)
        self.xr_btn = QPushButton("View in VR (6DoF)")
        self.xr_btn.clicked.connect(lambda: open_in_vr(self, self.path, self.viewer.scene) if self.path else None)
        self.xr_btn.setEnabled(False)
        head.addWidget(self.xr_btn)
        self.vr_btn = QPushButton("Make VR180 from this scene")
        self.vr_btn.clicked.connect(self.make_vr)
        self.vr_btn.setEnabled(False)
        head.addWidget(self.vr_btn)
        v.addLayout(head)
        self.viewer = SceneViewer()
        v.addWidget(self.viewer, 1)
        self.path: Path | None = None

    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open 3D scene", "", "Gaussian splats (*.ply *.splat)")
        if p:
            self.win.open_scene(Path(p))

    def load(self, path: Path) -> bool:
        from ..scene import SceneError, load_scene

        try:
            scene = load_scene(path)
        except (SceneError, OSError, ValueError) as e:
            QMessageBox.warning(self, APP_NAME, f"Cannot open {path.name}: {e}")
            return False
        self.viewer.set_scene(scene)
        self.path = path
        self.vr_btn.setEnabled(True)
        self.xr_btn.setEnabled(True)
        return True

    def make_vr(self):
        if not self.viewer.scene or not self.path:
            return
        out = QFileDialog.getExistingDirectory(self, "Save VR180 images to", str(self.path.parent))
        if not out:
            return
        scene, s = self.viewer.scene, self.win.settings
        self.vr_btn.setEnabled(False)
        self.vr_btn.setText("Rendering…")

        def work():
            from ..vr180 import StereoOptions, render_stereo, save_stereo_image

            head = scene.cameras[0].c2w if scene.cameras else np.eye(4)
            for layout, on in (("sbs", s.layout_sbs), ("tb", s.layout_tb)):
                if on:
                    so = StereoOptions(layout=layout, projection=s.projection, eye_resolution=s.eye_resolution,
                                       eye_separation_m=s.eye_separation_mm / 1000, fill_holes=s.fill_holes)
                    save_stereo_image(render_stereo(scene, so, head_c2w=head), so, Path(out), self.path.stem)

        def done():
            self.vr_btn.setEnabled(True)
            self.vr_btn.setText("Make VR180 from this scene")
            open_path(out)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        timer = QTimer(self)

        def poll():
            if not t.is_alive():
                timer.stop()
                done()

        timer.timeout.connect(poll)
        timer.start(200)
