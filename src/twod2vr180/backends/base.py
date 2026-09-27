"""Backend adapter contract.

The GUI never imports research code. Each upstream project is wrapped by an
adapter implementing prepare / can_run / estimate / run / export / cleanup,
and heavy work runs in a worker subprocess inside the backend's runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..hardware import HardwareReport
from ..paths import workers_dir

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

PREFIX = "@@2D2VR180 "
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Capability tags (docs/PLAN.md phase 5)
SCENE_STATIC = "scene_static"
OBJECT_CENTRIC = "object_centric"
HUMAN = "human"
ANIMAL = "animal"
DYNAMIC = "dynamic"
SINGLE_VIEW = "single_view"
MULTI_VIEW = "multi_view"
NOVEL_VIEW_COMPLETION = "novel_view_completion"


class BackendError(Exception):
    code = "backend_error"

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        if code:
            self.code = code


class BackendUnavailable(BackendError):
    code = "unavailable"


class JobCancelled(BackendError):
    code = "cancelled"


@dataclass
class Estimate:
    seconds: float | None
    vram_gb: float | None
    disk_gb: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class Availability:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    missing_models: list[str] = field(default_factory=list)
    missing_runtime: str | None = None


@dataclass
class JobInput:
    path: Path
    kind: str                         # "photo" | "video"
    work_dir: Path
    frames: list[Path] = field(default_factory=list)   # images handed to the backend
    video_kind: str | None = None     # static_scene | static_camera_dynamic | moving_camera
    fps: float | None = None
    analysis: dict | None = None


@dataclass
class BackendResult:
    backend: str
    scene_ply: Path | None = None
    splat: Path | None = None
    obj: Path | None = None
    frame_scenes: list[Path] = field(default_factory=list)   # per-frame geometry (npz) for 2.5D video
    cameras: list[dict] = field(default_factory=list)
    metric_scale: bool = False
    provenance_note: str = ""
    vram_peak_mib: int | None = None
    worker_env: dict = field(default_factory=dict)
    models_used: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        d["frame_scenes"] = [str(p) for p in self.frame_scenes]
        return d


@dataclass
class BackendContext:
    models: Any                  # ModelManager
    runtimes: Any                # RuntimeManager
    license_profile: str = "commercial"   # "commercial" | "personal_research"
    allow_cpu: bool = False               # run cpu-capable backends without an NVIDIA GPU (slow)


class Backend:
    id: str = "base"
    display_name: str = "Base"
    inputs: tuple[str, ...] = ()              # "photo", "video:static_scene", ...
    capabilities: frozenset[str] = frozenset()
    runtime_id: str | None = None
    maturity: str = "experimental"            # production | experimental | research | unsupported
    upstream: list[str] = []                  # names in config/upstream-lock.json
    commercial_use: bool = False
    cpu_capable: bool = False                 # worker can fall back to CPU inference
    description: str = ""

    # ---------------------------------------------------------------- contract
    def model_ids(self, options: dict) -> list[str]:
        return []

    def availability(self, ctx: BackendContext, hw: HardwareReport | None, options: dict | None = None) -> Availability:
        options = options or {}
        reasons: list[str] = []
        if self.maturity == "unsupported":
            return Availability(False, [self.description or "not supported in this build"])
        if not self.commercial_use and ctx.license_profile == "commercial":
            reasons.append("license: research/non-commercial only (switch license profile to "
                           "'personal/research' to enable)")
        missing_rt = None
        if self.runtime_id and not ctx.runtimes.is_installed(self.runtime_id):
            missing_rt = self.runtime_id
            reasons.append(f"runtime '{self.runtime_id}' not installed")
        missing = [m for m in self.model_ids(options) if m not in ctx.models.entries
                   or (not ctx.models.entries[m].extra.get("fetched_by_upstream")
                       and not ctx.models.is_installed(m))]
        if missing:
            reasons.append("models not installed: " + ", ".join(missing))
        if hw is not None and not (self.cpu_capable and getattr(ctx, "allow_cpu", False)):
            need = self.min_vram_gb(options)
            gpu = hw.best_gpu
            if need and (gpu is None or gpu.total_mib < need * 1024):
                reasons.append(f"needs an NVIDIA GPU with >= {need} GB VRAM"
                               + (f" (found {gpu.total_mib / 1024:.1f} GB)" if gpu else " (none found)"))
        return Availability(not reasons, reasons, missing, missing_rt)

    def min_vram_gb(self, options: dict) -> float | None:
        return None

    def prepare(self, ctx: BackendContext, options: dict) -> None:
        for m in self.model_ids(options):
            e = ctx.models.entries.get(m)
            if e is None or e.extra.get("fetched_by_upstream"):
                continue
            problems = ctx.models.verify(m, full_hash=False)
            if problems:
                raise BackendUnavailable(f"model {m}: " + "; ".join(problems), code="model_missing")

    def can_run(self, inp: JobInput, hw: HardwareReport | None, ctx: BackendContext,
                options: dict | None = None) -> tuple[bool, list[str]]:
        kind = inp.kind if inp.kind == "photo" else f"video:{inp.video_kind}"
        reasons = []
        if kind not in self.inputs:
            reasons.append(f"{self.id} does not handle {kind}")
        av = self.availability(ctx, hw, options)
        reasons += av.reasons
        return not reasons, reasons

    def estimate(self, inp: JobInput, hw: HardwareReport | None, options: dict) -> Estimate:
        return Estimate(None, self.min_vram_gb(options), None)

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress: ProgressFn,
            cancel: CancelFn) -> BackendResult:
        raise NotImplementedError

    def export(self, result: BackendResult, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        out = []
        for p in (result.scene_ply, result.splat, result.obj):
            if p and Path(p).exists():
                tgt = destination / Path(p).name
                if Path(p).resolve() != tgt.resolve():
                    shutil.copy2(p, tgt)
                out.append(tgt)
        return out

    def cleanup(self, inp: JobInput) -> None:
        tmp = inp.work_dir / "tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    def describe(self) -> dict:
        return {"id": self.id, "name": self.display_name, "inputs": list(self.inputs),
                "capabilities": sorted(self.capabilities), "runtime": self.runtime_id,
                "maturity": self.maturity, "commercial_use": self.commercial_use,
                "upstream": self.upstream, "description": self.description}


# -------------------------------------------------------------------- workers

def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=_NO_WINDOW)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def run_worker(python: Path | str, script: str, request: dict, work_dir: Path, progress: ProgressFn,
               cancel: CancelFn, env: dict | None = None, log: Callable[[str], None] | None = None,
               progress_range: tuple[float, float] = (0.0, 1.0), timeout_s: float | None = None) -> dict:
    """Launch a worker script in a runtime interpreter and follow its events.

    Returns {"result": {...}, "env": {...}}; raises BackendError on failure,
    JobCancelled on cancellation (the whole process tree is killed)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    req_path = work_dir / f"{Path(script).stem}_request.json"
    req_path.write_text(json.dumps(request, indent=2))
    log_path = work_dir / f"{Path(script).stem}.log"
    script_path = Path(script) if Path(script).is_absolute() else workers_dir() / script
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([str(python), str(script_path), str(req_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace",
                            cwd=str(work_dir), **kwargs)
    lo, hi = progress_range
    result: dict | None = None
    error: dict | None = None
    wenv: dict = {}
    t0 = time.time()
    stop = threading.Event()

    def _watch() -> None:
        while not stop.wait(0.25):
            if cancel() or (timeout_s and time.time() - t0 > timeout_s):
                _kill_tree(proc)
                return

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        assert proc.stdout is not None
        with open(log_path, "a", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                if not line.startswith(PREFIX):
                    if log:
                        log(line.rstrip())
                    continue
                try:
                    ev = json.loads(line[len(PREFIX):])
                except json.JSONDecodeError:
                    continue
                kind = ev.get("event")
                if kind == "progress":
                    progress(lo + (hi - lo) * float(ev.get("value", 0)), ev.get("message", ""))
                elif kind == "log" and log:
                    log(ev.get("message", ""))
                elif kind == "env":
                    wenv = ev
                elif kind == "result":
                    result = ev
                elif kind == "error":
                    error = ev
        rc = proc.wait()
    finally:
        stop.set()
    if cancel():
        raise JobCancelled("Job cancelled by user.")
    if timeout_s and time.time() - t0 > timeout_s and result is None:
        raise BackendError(f"Worker timed out after {timeout_s:.0f}s", code="timeout")
    if error:
        raise BackendError(error.get("message", "worker error"), code=error.get("code", "upstream_error"))
    if rc != 0 or result is None:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:] if log_path.exists() else ""
        raise BackendError(f"Worker exited with code {rc} without a result. Log tail:\n{tail}",
                           code="worker_crash")
    return {"result": result, "env": wenv}
