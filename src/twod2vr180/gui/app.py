"""2D2VR180 desktop application (PySide6).

The GUI process only orchestrates: jobs run on a QThread, heavy inference runs
in backend worker processes, and the viewer uses the numpy reference renderer
so it works on any machine.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..media import IMAGE_EXTS, VIDEO_EXTS
from .viewer import SceneViewer

PROVENANCE_HELP = (
    "<b>Observed</b>: reconstructed from several views. <b>Inferred</b>: seen in the input but depth predicted "
    "by a network. <b>Generative</b>: invented by a completion model. <b>Unknown</b>: no data (black in VR180).")


class DropZone(QLabel):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setWordWrap(True)
        self.setStyleSheet("QLabel{border:2px dashed #888;border-radius:8px;padding:12px;font-size:14px}")
        self.setText("Drop a photo (JPG/PNG/WebP) or video (MP4/MOV/MKV) here\n— or click to browse —")

    def mousePressEvent(self, ev):  # noqa: N802
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS | VIDEO_EXTS))
        path, _ = QFileDialog.getOpenFileName(self, "Open photo or video", "", f"Media ({exts})")
        if path:
            self.fileDropped.emit(path)

    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):  # noqa: N802
        urls = ev.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())


class JobWorker(QObject):
    event = Signal(dict)
    done = Signal(dict)

    def __init__(self, job, runner):
        super().__init__()
        self.job = job
        self.runner = runner

    def run(self):
        rep = self.runner.run(self.job, self.event.emit)
        self.done.emit(rep)


class TaskWorker(QObject):
    """Generic background task (model download / runtime install)."""
    message = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel_flag = threading.Event()

    def run(self):
        try:
            self.fn(self.message.emit, self.cancel_flag.is_set)
            self.finished.emit(True, "")
        except Exception as e:  # noqa: BLE001 - surfaced to the user
            self.finished.emit(False, str(e))


def _np_to_pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from ..backends.base import BackendContext
        from ..models import ModelManager
        from ..paths import app_paths
        from ..runtimes import RuntimeManager

        self.paths = app_paths().ensure()
        self.ctx = BackendContext(ModelManager(), RuntimeManager(), "personal_research")
        self.hw = None
        self.input_path: str | None = None
        self.job = None
        self.thread: QThread | None = None
        self.tasks: list[tuple[QThread, TaskWorker]] = []
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1360, 860)
        self._build()
        self.render_diagnostics()
        QTimer.singleShot(50, self.refresh_hardware)

    # ------------------------------------------------------------------ layout
    def _build(self):
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        self.drop = DropZone()
        self.drop.fileDropped.connect(self.set_input)
        lv.addWidget(self.drop)

        box = QGroupBox("Generate")
        form = QFormLayout(box)
        self.mode = QComboBox()
        self.mode.addItems(["Auto", "Quality", "Fast"])
        form.addRow("Mode", self.mode)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Automatic", None)
        from ..backends import all_backends

        for b in all_backends():
            if b.maturity != "unsupported":
                self.backend_combo.addItem(b.display_name, b.id)
        form.addRow("Backend", self.backend_combo)
        self.profile = QComboBox()
        self.profile.addItem("Personal / research use", "personal_research")
        self.profile.addItem("Commercial use (restricts models)", "commercial")
        form.addRow("License profile", self.profile)
        self.vr = QCheckBox("Generate VR180")
        self.vr.setChecked(True)
        form.addRow(self.vr)
        self.sbs = QCheckBox("Side-by-side")
        self.sbs.setChecked(True)
        self.tb = QCheckBox("Top/Bottom")
        self.tb.setChecked(True)
        lay = QHBoxLayout()
        lay.addWidget(self.sbs)
        lay.addWidget(self.tb)
        form.addRow("Layouts", lay)
        self.projection = QComboBox()
        self.projection.addItem("VR180 (half-equirectangular)", "equirect180")
        self.projection.addItem("Flat 3D (source field of view)", "flat")
        form.addRow("Projection", self.projection)
        self.eye_res = QSpinBox()
        self.eye_res.setRange(512, 4096)
        self.eye_res.setSingleStep(256)
        self.eye_res.setValue(2048)
        form.addRow("Eye resolution", self.eye_res)
        lv.addWidget(box)

        row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.clicked.connect(self.generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.generate_btn, 3)
        row.addWidget(self.cancel_btn, 1)
        lv.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        lv.addWidget(self.progress)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        lv.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        lv.addWidget(self.log, 1)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_output)
        lv.addWidget(self.open_btn)

        self.tabs = QTabWidget()
        self.viewer = SceneViewer()
        self.tabs.addTab(self.viewer, "3D viewer")
        self.vr_label = QLabel("VR180 preview appears here after generation.")
        self.vr_label.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.vr_label, "VR180")
        self.diag = QTextBrowser()
        self.diag.setOpenExternalLinks(True)
        self.tabs.addTab(self.diag, "Diagnostics")
        self.tabs.addTab(self._build_manager(), "Models && runtimes")
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([420, 940])
        self.setCentralWidget(split)

        m = self.menuBar().addMenu("&File")
        act = QAction("Open 3D scene (.ply/.splat)…", self)
        act.triggered.connect(self.open_scene)
        m.addAction(act)
        act2 = QAction("Open jobs folder", self)
        act2.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.jobs))))
        m.addAction(act2)

    def _build_manager(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("<b>Runtimes</b> — isolated GPU environments, downloaded once (several GB)."))
        self.rt_table = QTableWidget(0, 4)
        self.rt_table.setHorizontalHeaderLabels(["Runtime", "Status", "Size", "Description"])
        self.rt_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        v.addWidget(self.rt_table)
        rb = QHBoxLayout()
        b1 = QPushButton("Install selected runtime")
        b1.clicked.connect(self.install_runtime)
        rb.addWidget(b1)
        b2 = QPushButton("Remove selected runtime")
        b2.clicked.connect(self.remove_runtime)
        rb.addWidget(b2)
        v.addLayout(rb)
        v.addWidget(QLabel("<b>Models</b> — downloaded from their original hosts; never bundled."))
        self.m_table = QTableWidget(0, 5)
        self.m_table.setHorizontalHeaderLabels(["Model", "Status", "Size", "License", "Commercial"])
        self.m_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        v.addWidget(self.m_table)
        mb = QHBoxLayout()
        b3 = QPushButton("Download selected model…")
        b3.clicked.connect(self.download_model)
        mb.addWidget(b3)
        b4 = QPushButton("Verify")
        b4.clicked.connect(self.verify_model)
        mb.addWidget(b4)
        b5 = QPushButton("Delete")
        b5.clicked.connect(self.delete_model)
        mb.addWidget(b5)
        v.addLayout(mb)
        self.refresh_manager()
        return w

    # ------------------------------------------------------------------ state
    def refresh_hardware(self):
        from ..hardware import format_report, probe

        self.hw = probe()
        self.render_diagnostics()
        if not self.hw.gpus:
            self.status.setText("No NVIDIA GPU detected — reconstruction backends are unavailable. "
                                "You can still open and view existing 3D scenes.")

    def render_diagnostics(self, report: dict | None = None):
        from ..backends import all_backends
        from ..hardware import format_report

        html = [f"<h3>{APP_NAME} {__version__}</h3>"]
        if self.hw:
            html.append("<h4>Hardware</h4><pre>" + format_report(self.hw) + "</pre>")
        html.append("<h4>Backends</h4><table border=1 cellspacing=0 cellpadding=4>"
                    "<tr><th>Backend</th><th>Maturity</th><th>Available</th><th>Reason</th></tr>")
        for b in all_backends():
            av = b.availability(self.ctx, self.hw, {"mode": self.mode.currentText().lower()})
            html.append(f"<tr><td>{b.display_name}</td><td>{b.maturity}</td><td>{'yes' if av.ok else 'no'}</td>"
                        f"<td>{'<br>'.join(av.reasons)}</td></tr>")
        html.append("</table>")
        if report:
            cov = report.get("coverage", {})
            html.append("<h4>Last job — geometry provenance</h4><p>" + PROVENANCE_HELP + "</p>")
            html.append("<pre>" + json.dumps(cov, indent=2) + "</pre>")
            vr = (report.get("outputs") or {}).get("vr180", {})
            for st in vr.get("stills", []):
                md = st["metadata"]
                html.append(f"<p><b>{Path(st['image']).name}</b>: content FOV "
                            f"{md['content_fov_deg']['horizontal_deg']}°×{md['content_fov_deg']['vertical_deg']}°, "
                            f"full VR180: {md['is_full_vr180']}<br>{md.get('honesty_note', '')}</p>")
            html.append("<h4>Selection</h4><pre>" + json.dumps(report.get("selection"), indent=2) + "</pre>")
            html.append("<h4>Validation</h4><pre>" + json.dumps(report.get("validation"), indent=2) + "</pre>")
            if report.get("warnings"):
                html.append("<h4>Warnings</h4><ul>" + "".join(f"<li>{w}</li>" for w in report["warnings"]) + "</ul>")
            if report.get("error"):
                html.append(f"<h4>Error</h4><p style='color:#c33'>[{report['error']['code']}] "
                            f"{report['error']['message']}</p>")
        self.diag.setHtml("".join(html))

    def refresh_manager(self):
        rm, mm = self.ctx.runtimes, self.ctx.models
        self.rt_table.setRowCount(0)
        for rid in rm.specs:
            s = rm.status(rid)
            r = self.rt_table.rowCount()
            self.rt_table.insertRow(r)
            for c, val in enumerate([rid, "installed" if s["installed"] else "not installed",
                                     f"~{s['approx_size_gb']} GB", f"{s['description']} ({s['manifest_status']})"]):
                self.rt_table.setItem(r, c, QTableWidgetItem(val))
        self.m_table.setRowCount(0)
        for mid in mm.entries:
            s = mm.status(mid)
            r = self.m_table.rowCount()
            self.m_table.insertRow(r)
            state = "installed" if s["installed"] else (
                "fetched by runtime" if mm.entries[mid].extra.get("fetched_by_upstream") else "not installed")
            size = f"{s['size_bytes'] / 2**30:.2f} GB" if s["size_bytes"] else "?"
            for c, val in enumerate([mid, state, size, s["license"], "yes" if s["commercial_use"] else "no"]):
                self.m_table.setItem(r, c, QTableWidgetItem(val))

    # ------------------------------------------------------------------ input / job
    def set_input(self, path: str):
        p = Path(path)
        ext = p.suffix.lower()
        if ext in (".ply", ".splat"):
            self.load_scene(p)
            return
        if ext not in IMAGE_EXTS | VIDEO_EXTS:
            QMessageBox.warning(self, APP_NAME, f"Unsupported file type: {ext}")
            return
        self.input_path = str(p)
        self.drop.setText(f"{p.name}\n({'video' if ext in VIDEO_EXTS else 'photo'})")
        self.status.setText("Ready to generate.")

    def _options(self):
        from ..jobs import JobOptions

        layouts = [x for x, cb in (("sbs", self.sbs), ("tb", self.tb)) if cb.isChecked()] or ["sbs"]
        return JobOptions(mode=self.mode.currentText().lower(), backend=self.backend_combo.currentData(),
                          vr180=self.vr.isChecked(), layouts=layouts, projection=self.projection.currentData(),
                          eye_resolution=self.eye_res.value(), license_profile=self.profile.currentData())

    def generate(self):
        if not self.input_path:
            QMessageBox.information(self, APP_NAME, "Drop a photo or video first.")
            return
        from ..jobs import Job, JobRunner

        self.job = Job(Path(self.input_path), self._options())
        runner = JobRunner(self.ctx, self.hw)
        self.thread = QThread()
        self.worker = JobWorker(self.job, runner)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.event.connect(self.on_event)
        self.worker.done.connect(self.on_done)
        self.worker.done.connect(self.thread.quit)
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.log.clear()
        self.progress.setValue(0)
        self.thread.start()

    def cancel(self):
        if self.job:
            self.job.cancel()
            self.status.setText("Cancelling…")

    def on_event(self, ev: dict):
        k = ev.get("event")
        if k == "progress":
            self.progress.setValue(int(ev.get("value", 0) * 1000))
            self.status.setText(ev.get("message", ""))
            self.log.appendPlainText(f"[{ev.get('value', 0) * 100:5.1f}%] {ev.get('message', '')}")
        elif k == "log":
            self.log.appendPlainText(ev.get("message", ""))

    def on_done(self, report: dict):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        st = report.get("status")
        if st == "succeeded":
            self.status.setText("Done. " + ("  ".join(report.get("warnings", [])[:2])))
            ply = report["outputs"]["scene_ply"]
            self.load_scene(Path(ply))
            stills = report["outputs"].get("vr180", {}).get("stills", [])
            if stills:
                pm = QPixmap(stills[0]["image"])
                self.vr_label.setPixmap(pm.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            err = report.get("error", {})
            self.status.setText(f"{st}: {err.get('message', '')}")
            if st == "failed":
                QMessageBox.warning(self, APP_NAME, f"The job failed [{err.get('code')}]:\n\n{err.get('message')}"
                                    "\n\nDetails are in the Diagnostics tab and run_report.json.")
        self.render_diagnostics(report)

    def open_output(self):
        if self.job:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.job.dir)))

    def open_scene(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open 3D scene", "", "Gaussian splats (*.ply *.splat)")
        if path:
            self.load_scene(Path(path))

    def load_scene(self, path: Path):
        from ..scene import SceneError, load_scene

        try:
            scene = load_scene(path)
        except (SceneError, OSError, ValueError) as e:
            QMessageBox.warning(self, APP_NAME, f"Cannot open {path.name}: {e}")
            return
        self.viewer.set_scene(scene)
        self.tabs.setCurrentWidget(self.viewer)

    # ------------------------------------------------------------------ manager actions
    def _selected(self, table: QTableWidget) -> str | None:
        r = table.currentRow()
        return table.item(r, 0).text() if r >= 0 else None

    def _start_task(self, fn, label: str):
        th = QThread()
        wk = TaskWorker(fn)
        wk.moveToThread(th)
        th.started.connect(wk.run)
        wk.message.connect(lambda m: self.log.appendPlainText(m))
        wk.finished.connect(lambda ok, msg: self._task_done(ok, msg, label))
        wk.finished.connect(th.quit)
        self.tasks.append((th, wk))
        self.status.setText(f"{label}…")
        th.start()

    def _task_done(self, ok: bool, msg: str, label: str):
        self.status.setText(f"{label}: {'done' if ok else 'FAILED'} {msg}")
        if not ok:
            QMessageBox.warning(self, APP_NAME, f"{label} failed:\n{msg}")
        self.refresh_manager()
        self.render_diagnostics()

    def install_runtime(self):
        rid = self._selected(self.rt_table)
        if not rid:
            return
        spec = self.ctx.runtimes.specs[rid]
        if QMessageBox.question(self, APP_NAME, f"Download and install runtime '{rid}' (~{spec.approx_size_gb} GB)?\n\n"
                                f"{spec.description}\nStatus: {spec.status}") != QMessageBox.Yes:
            return
        self._start_task(lambda log, cancel: self.ctx.runtimes.install(rid, log, cancel), f"Installing {rid}")

    def remove_runtime(self):
        rid = self._selected(self.rt_table)
        if rid and QMessageBox.question(self, APP_NAME, f"Remove runtime '{rid}'?") == QMessageBox.Yes:
            self.ctx.runtimes.remove(rid)
            self.refresh_manager()

    def download_model(self):
        mid = self._selected(self.m_table)
        if not mid:
            return
        mm = self.ctx.models
        e = mm.entries[mid]
        if e.extra.get("fetched_by_upstream"):
            QMessageBox.information(self, APP_NAME, "This model is fetched automatically by its backend runtime.")
            return
        size = f"{e.size_bytes / 2**30:.2f} GB" if e.size_bytes else "unknown size"
        msg = (f"<b>{e.display_name}</b><br>Source: {e.source}<br>Size: {size}<br>"
               f"License: {e.license}<br><a href='{e.license_url}'>{e.license_url}</a><br>"
               f"Commercial use: {'yes' if e.commercial_use else 'NO'} · Redistributable: "
               f"{'yes' if e.redistributable else 'no'}<br>Expected SHA256: {e.files[0].sha256 or 'recorded on first download'}"
               f"<br><br>{e.notes}<br><br>Do you accept this license and want to download?")
        if QMessageBox.question(self, "Model license", msg) != QMessageBox.Yes:
            return
        mm.accept_license(mid)

        def task(log, cancel):
            last = [0]

            def prog(done, total):
                if done - last[0] > 50 * 2**20:
                    last[0] = done
                    log(f"{mid}: {done / 2**20:.0f} MiB" + (f" / {total / 2**20:.0f}" if total else ""))

            mm.download(mid, progress=prog, cancel=cancel, hf_token=os.environ.get("HF_TOKEN"))

        self._start_task(task, f"Downloading {mid}")

    def verify_model(self):
        mid = self._selected(self.m_table)
        if mid:
            probs = self.ctx.models.verify(mid)
            QMessageBox.information(self, APP_NAME, "OK — hash verified." if not probs else "\n".join(probs))

    def delete_model(self):
        mid = self._selected(self.m_table)
        if mid and QMessageBox.question(self, APP_NAME, f"Delete model '{mid}'?") == QMessageBox.Yes:
            self.ctx.models.delete(mid)
            self.refresh_manager()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    if argv:
        w.set_input(argv[0])
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
