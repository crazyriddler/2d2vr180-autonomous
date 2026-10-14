"""Background work for the GUI: a sequential job queue and a component
installer. Work runs on plain Python threads; results reach the UI through Qt
signals (queued across threads automatically)."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal


class JobQueue(QObject):
    event = Signal(dict)     # every job event, plus {"event": "queued"/"started"}
    changed = Signal()

    def __init__(self, ctx, hw_getter):
        super().__init__()
        self.ctx = ctx
        self.hw_getter = hw_getter
        self.jobs: list = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def add(self, path, options) -> object:
        """``path``: one photo/video, or a list of photos combined into one multi-view scene."""
        from ..jobs import Job

        job = Job([Path(p) for p in path] if isinstance(path, (list, tuple)) else Path(path), options)
        with self._lock:
            self.jobs.append(job)
        self.event.emit({"event": "queued", "job": job.id})
        self.changed.emit()
        self._ensure_running()
        return job

    def pending(self) -> list:
        return [j for j in self.jobs if j.state in ("queued", "running")]

    def cancel(self, job_id: str) -> None:
        for j in self.jobs:
            if j.id == job_id and j.state in ("queued", "running"):
                j.cancel()
                if j.state == "queued":
                    j.state = "cancelled"
        self.changed.emit()

    def cancel_all(self) -> None:
        for j in self.pending():
            self.cancel(j.id)

    def _ensure_running(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="job-queue", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        from ..jobs import JobRunner

        while True:
            with self._lock:
                nxt = next((j for j in self.jobs if j.state == "queued"), None)
            if nxt is None:
                return
            if nxt.cancelled:
                nxt.state = "cancelled"
                self.changed.emit()
                continue
            self.event.emit({"event": "started", "job": nxt.id})
            self.changed.emit()
            try:
                JobRunner(self.ctx, self.hw_getter()).run(nxt, self.event.emit)
            except Exception as e:  # noqa: BLE001 - JobRunner already reports; never kill the queue
                nxt.state = "failed"
                self.event.emit({"event": "finished", "job": nxt.id, "status": "failed",
                                 "error": {"code": "internal_error", "message": f"{e}\n{traceback.format_exc()}"}})
            self.changed.emit()


class ComponentInstaller(QObject):
    progress = Signal(str, float, str)   # component id, fraction (-1 = indeterminate), message
    finished = Signal(str, bool, str)    # component id, ok, message
    idle = Signal()

    def __init__(self, ctx, token_getter):
        super().__init__()
        self.ctx = ctx
        self.token_getter = token_getter
        self.queue: list = []
        self.current: str | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def enqueue(self, components: list) -> None:
        for c in components:
            if c.id != self.current and all(q.id != c.id for q in self.queue):
                self.queue.append(c)
                self.progress.emit(c.id, -1.0, "waiting…")
        if not (self._thread and self._thread.is_alive()):
            self._cancel.clear()
            self._thread = threading.Thread(target=self._loop, name="installer", daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        for c in self.queue:
            self.finished.emit(c.id, False, "cancelled")
        self.queue.clear()

    def busy(self) -> bool:
        return bool(self.current or self.queue)

    def _loop(self) -> None:
        from ..components import component_status, get_component, install_component

        while self.queue:
            c = self.queue.pop(0)
            self.current = c.id
            try:
                missing = [d for d in c.requires if not component_status(self.ctx, get_component(d))["installed"]]
                if missing:
                    raise RuntimeError(f"requires {', '.join(missing)} to be installed first")
                install_component(self.ctx, c,
                                  lambda f, m, cid=c.id: self.progress.emit(cid, -1.0 if f is None else float(f), m),
                                  self._cancel.is_set, self.token_getter())
                self.finished.emit(c.id, True, "installed")
            except Exception as e:  # noqa: BLE001 - surfaced in the UI
                self.finished.emit(c.id, False, str(e))
            finally:
                self.current = None
        self.idle.emit()
