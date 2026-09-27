"""2D2VR180 desktop application (PySide6).

The GUI process only orchestrates: jobs run on a background queue, heavy
inference runs in isolated backend worker processes, and the viewer uses the
numpy reference renderer so it works on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..paths import resource_root

STYLE = """
QWidget { font-size: 13px; }
QLabel#h1 { font-size: 22px; font-weight: 600; margin: 4px 0 8px 0; }
QLabel#note { color: #a9adb8; }
QLabel#banner { background: #5a4410; color: #ffe7a8; padding: 8px; border-radius: 6px; }
QLabel#dropzone { border: 2px dashed #5b6273; border-radius: 12px; padding: 16px; color: #d6d9e0; background: #22252c; }
QLabel#dropzone:hover { border-color: #4f8cff; }
QFrame#card { background: #262a32; border: 1px solid #343a46; border-radius: 10px; padding: 8px; }
QPushButton { padding: 6px 14px; border-radius: 6px; background: #343a46; border: 1px solid #434a58; }
QPushButton:hover { background: #3d4452; }
QPushButton:disabled { color: #6b7080; background: #2a2e36; }
QPushButton#primary { background: #2f6fed; border-color: #2f6fed; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3b7bff; }
QListWidget#nav { background: #1b1d22; border: none; font-size: 14px; }
QListWidget#nav::item { padding: 12px 14px; }
QListWidget#nav::item:selected { background: #2f6fed; color: white; border-radius: 6px; }
QGroupBox { border: 1px solid #343a46; border-radius: 8px; margin-top: 12px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QProgressBar { border: 1px solid #434a58; border-radius: 5px; text-align: center; height: 16px; }
QProgressBar::chunk { background: #2f6fed; border-radius: 4px; }
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.Window, "#1f2228"), (QPalette.WindowText, "#e6e8ee"), (QPalette.Base, "#181a1f"),
                      (QPalette.AlternateBase, "#23262d"), (QPalette.Text, "#e6e8ee"), (QPalette.Button, "#343a46"),
                      (QPalette.ButtonText, "#e6e8ee"), (QPalette.Highlight, "#2f6fed"),
                      (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#23262d"),
                      (QPalette.ToolTipText, "#e6e8ee"), (QPalette.Link, "#7aa7ff")):
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#6b7080"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6b7080"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def app_icon() -> QIcon:
    p = resource_root() / "assets" / "icon.png"
    return QIcon(str(p)) if p.exists() else QIcon()


class WelcomeDialog(QDialog):
    def __init__(self, parent, hw):
        super().__init__(parent)
        self.setWindowTitle(f"Welcome to {APP_NAME}")
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        t = QLabel(f"<h2>Welcome to {APP_NAME}</h2>")
        v.addWidget(t)
        if hw and hw.gpus:
            g = hw.best_gpu
            gpu = f"Detected <b>{g.name}</b> with {g.total_mib / 1024:.0f} GB of video memory."
        else:
            gpu = ("<b>No NVIDIA GPU was detected.</b> You can still view 3D files; reconstruction needs an NVIDIA "
                   "GPU (or CPU mode in Settings, which is very slow).")
        body = QLabel(
            f"<p>{gpu}</p><p>Turn photos and videos into explorable 3D scenes, Gaussian splats, meshes and VR180 "
            "stereo — entirely on this computer.</p><p>Before the first run, the app downloads its AI engine "
            "(about 5–6 GB) and models (0.1–1.5 GB each) from their original publishers. You will see each "
            "model's licence before it is downloaded.</p>")
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        v.addWidget(body)
        row = QHBoxLayout()
        row.addStretch(1)
        later = QPushButton("Later")
        later.clicked.connect(self.reject)
        go = QPushButton("Install recommended components…")
        go.setObjectName("primary")
        go.clicked.connect(self.accept)
        row.addWidget(later)
        row.addWidget(go)
        v.addLayout(row)


class MainWindow(QMainWindow):
    PAGES = ["Create", "Results", "3D viewer", "Components", "Settings", "Diagnostics"]

    def __init__(self):
        super().__init__()
        from ..backends.base import BackendContext
        from ..models import ModelManager
        from ..paths import app_paths
        from ..runtimes import RuntimeManager
        from ..settings import Settings
        from .pages import ComponentsPage, CreatePage, DiagnosticsPage, ResultsPage, SettingsPage, ViewerPage
        from .workers import ComponentInstaller, JobQueue

        self.paths = app_paths().ensure()
        self.settings = Settings.load()
        self.ctx = BackendContext(ModelManager(), RuntimeManager(), self.settings.license_profile,
                                  self.settings.allow_cpu)
        self.hw = None
        self.queue = JobQueue(self.ctx, lambda: self.hw)
        self.installer = ComponentInstaller(self.ctx, lambda: self.settings.hf_token)
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1380, 880)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        brand = QListWidgetItem(f"{APP_NAME}")
        brand.setFlags(Qt.NoItemFlags)
        brand.setIcon(app_icon())
        self.nav.setIconSize(QSize(28, 28))
        self.nav.addItem(brand)
        for p in self.PAGES:
            self.nav.addItem(p)
        h.addWidget(self.nav)
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.create = CreatePage(self)
        self.results = ResultsPage(self)
        self.viewer_page = ViewerPage(self)
        self.components = ComponentsPage(self)
        self.settings_page = SettingsPage(self)
        self.diagnostics = DiagnosticsPage(self)
        for w in (self.create, self.results, self.viewer_page, self.components, self.settings_page, self.diagnostics):
            wrap = QWidget()
            lv = QVBoxLayout(wrap)
            lv.setContentsMargins(18, 12, 18, 12)
            lv.addWidget(w)
            self.stack.addWidget(wrap)
        self.nav.currentRowChanged.connect(lambda r: self.stack.setCurrentIndex(max(r - 1, 0)))
        self.nav.setCurrentRow(1)

        self.create.openComponents.connect(lambda: self.go("Components"))
        self.results.exploreScene.connect(lambda p: self.open_scene(Path(p)))
        self.components.changed.connect(self._components_changed)
        self.settings_page.changed.connect(self._components_changed)
        self.queue.event.connect(self._job_event)
        self.queue.changed.connect(self.create.refresh_queue)
        self.statusBar().showMessage("Ready.")
        QTimer.singleShot(30, self.refresh_hardware)

    # ------------------------------------------------------------------
    def go(self, page: str) -> None:
        self.nav.setCurrentRow(self.PAGES.index(page) + 1)

    def refresh_hardware(self) -> None:
        from ..hardware import probe

        self.hw = probe()
        self._components_changed()
        if not self.settings.setup_completed:
            QTimer.singleShot(50, self.first_run)

    def first_run(self) -> None:
        self.settings.setup_completed = True
        self.settings.save()
        dlg = WelcomeDialog(self, self.hw)
        if dlg.exec() == QDialog.Accepted:
            self.go("Components")
            self.components.install_recommended()

    def _components_changed(self) -> None:
        self.create.refresh_backends()
        self.diagnostics.render()

    def _job_event(self, ev: dict) -> None:
        self.create.on_event(ev)
        if ev.get("event") == "progress":
            self.statusBar().showMessage(f"{ev.get('value', 0) * 100:.0f}% — {ev.get('message', '')}")
        if ev.get("event") == "finished":
            job = next((j for j in self.queue.jobs if j.id == ev.get("job")), None)
            if job and job.report:
                self.diagnostics.last_report = job.report
                self.diagnostics.render()
            self.results.refresh()
            st = ev.get("status")
            self.statusBar().showMessage(f"Job {st}.")
            if st == "succeeded" and not self.queue.pending():
                self.go("Results")
            elif st == "failed":
                err = ev.get("error") or {}
                QMessageBox.warning(self, APP_NAME, f"The job failed [{err.get('code')}]:\n\n{err.get('message')}\n\n"
                                    "Details are on the Diagnostics page and in the job's run_report.json.")

    def open_scene(self, path: Path) -> None:
        if self.viewer_page.load(path):
            self.go("3D viewer")

    def closeEvent(self, ev):  # noqa: N802
        busy = self.queue.pending() or self.installer.busy()
        if busy and QMessageBox.question(self, APP_NAME, "Work is still running. Cancel it and quit?") != QMessageBox.Yes:
            ev.ignore()
            return
        self.queue.cancel_all()
        self.installer.cancel()
        ev.accept()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    apply_theme(app)
    app.setWindowIcon(app_icon())
    w = MainWindow()
    if argv:
        p = Path(argv[0])
        if p.suffix.lower() in (".ply", ".splat"):
            w.open_scene(p)
        else:
            w.create.add_files([str(p)])
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
