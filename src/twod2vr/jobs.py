"""Job system: ingest → select → reconstruct (Real 3D for a photo, a splat video for a video) → export →
validate → run_report.json.

A report is written for every job, including failed and cancelled ones.
Jobs run in the calling thread (CLI) or a worker thread (GUI); cancellation is
cooperative in-process and kills the worker process tree out-of-process.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .backends.base import BackendContext, BackendError, BackendResult, JobCancelled, JobInput
from .hardware import HardwareReport, probe
from .media import (MediaError, classify_path, exif_hfov_deg, extract_audio, extract_frames_fps, find_ffmpeg,
                    load_image, probe_video)
from .paths import app_paths, config_dir
from .selector import select

EventFn = Callable[[dict], None]


@dataclass
class JobOptions:
    mode: str = "auto"                   # auto | quality | fast
    backend: str | None = None           # force a backend id
    license_profile: str = "personal_research"
    output_dir: str | None = None        # copy final outputs here too
    allow_cpu: bool = False              # allow CPU inference for cpu-capable backends (slow)
    real3d_fill: bool = True             # photo: AI-paint what the photo hides from moved views
    ipd_mm: float = 63.0                 # the people's eyes are scaled to this distance (real-world scale)
    photo_detail: bool = False           # photos: maximum detail (always on in the Quality mode)
    video_fps: float = 0.0               # splat video: frames per second (0 = the video's own, up to 24)
    video_seconds: float = 0.0           # splat video: how much of the video (0 = all of it)
    video_start_s: float = 0.0           # splat video: where to start in the video

    @classmethod
    def from_dict(cls, d: dict) -> "JobOptions":
        """Options from a saved dict, ignoring keys of older versions."""
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def expand_inputs(paths) -> list[Path]:
    """A file, several files, or a folder of photos → list of input files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:
            out.append(p)
    return out


class Job:
    def __init__(self, input_path, options: JobOptions | None = None, jobs_root: Path | None = None):
        """``input_path``: one photo/video, a list of photos of the same scene, or a folder of photos."""
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.inputs = expand_inputs(input_path) or [Path(input_path)]
        self.input = self.inputs[0]
        self.options = options or JobOptions()
        self.dir = (jobs_root or app_paths().jobs) / self.id
        self.state = "queued"
        self.progress = 0.0
        self.message = ""
        self.report: dict | None = None
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()


def _sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def upstream_commits(names: list[str]) -> dict:
    try:
        lock = json.loads((config_dir() / "upstream-lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for r in lock.get("repositories", []):
        if r["name"] in names:
            out[r["name"]] = {"url": r.get("url"), "commit": r.get("commit"), "license": r.get("code_license")}
    return out


def write_export_readme(export: Path, report: dict) -> None:
    """Plain-language guide to the files a job produced (README.txt in the export folder)."""
    out = report.get("outputs") or {}
    lines = [f"2D2VR results for {Path((report.get('input') or {}).get('path', '')).name}", ""]
    if out.get("splv"):
        lines += ["SPLAT VIDEO (play it in the app's 3D viewer, or with 'View in VR')",
                  "  scene.splv   The video in 3D: every frame is a Gaussian splat, with the sound. Plays in 2D2VR",
                  "               (3D viewer and 'View in VR', which streams it to the headset's browser).",
                  "  scene.ply    The first frame as a still splat, standard 3DGS format (SuperSplat, Postshot,",
                  "               Blender and most splat viewers).",
                  "  scene.splat  The first frame in the compact web format."]
    else:
        lines += ["3D SCENE (explore it in the app's 3D viewer, or with 'View in VR')",
                  "  scene.ply    Gaussian splat, standard 3DGS format. Opens in SuperSplat (superspl.at/editor),",
                  "               Postshot, Polycam, Luma, Blender (3DGS add-ons) and most splat viewers.",
                  "  scene.splat  Same splat in the compact web format (antimatter15 / many web viewers).",
                  "  views_sheet.png  The splat seen from a few moved positions (a quick check of the 3D)."]
    if out.get("obj"):
        lines += ["  scene.obj    Triangle mesh with scene.mtl + scene_texture.png (Blender, MeshLab, 3D printing…)."]
    lines += ["  _2d2vr\\   Metadata the app uses (cameras, which parts are observed/inferred). Keep it next to",
              "               scene.ply if you want to reopen the scene in 2D2VR; other programs ignore it.", "",
              "WALK AROUND THE SPLAT IN VR (6DoF)",
              "  In the app: Results → 'View in VR'. Connect the headset to this PC first (Quest Link,",
              "  Air Link, Virtual Desktop or SteamVR), then press ENTER VR in the page that opens.",
              "  Controllers: A / X smaller, B / Y bigger (if the people do not look life-size), thumbstick",
              "  click resets; trigger plays / pauses a splat video.", "",
              "Details of this run: ../run_report.json"]
    try:
        (export / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


class JobRunner:
    def __init__(self, ctx: BackendContext, hardware: HardwareReport | None = None):
        self.ctx = ctx
        self.hw = hardware
        self._hw: HardwareReport | None = hardware

    # ------------------------------------------------------------------ run
    def run(self, job: Job, on_event: EventFn | None = None) -> dict:
        job.dir.mkdir(parents=True, exist_ok=True)
        opts = job.options
        self.ctx.license_profile = opts.license_profile
        self.ctx.allow_cpu = opts.allow_cpu
        timings: dict[str, float] = {}
        warnings: list[str] = []
        logs: list[str] = []
        report: dict = {
            "schema": "2d2vr.run_report/1",
            "app": {"name": "2D2VR", "version": __version__},
            "job_id": job.id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "options": asdict(opts),
            "status": "running",
        }
        log_file = open(job.dir / "job.log", "a", encoding="utf-8")

        def emit(kind: str, **kw) -> None:
            ev = {"event": kind, "job": job.id, **kw}
            if kind == "progress":
                job.progress = float(kw.get("value", job.progress))
                job.message = kw.get("message", job.message)
            if kind in ("log", "warning", "progress"):
                log_file.write(f"[{time.strftime('%H:%M:%S')}] {kind}: {kw.get('message', '')}\n")
                log_file.flush()
            if on_event:
                on_event(ev)

        def log(msg: str) -> None:
            logs.append(msg)
            emit("log", message=msg)

        def check_cancel() -> None:
            if job.cancelled:
                raise JobCancelled("Job cancelled by user.")

        stage_t = [time.time()]

        def stage(name: str) -> None:
            now = time.time()
            timings[name] = round(now - stage_t[0], 3)
            stage_t[0] = now

        result: BackendResult | None = None
        job.state = "running"
        final_state = "failed"
        try:
            # ---------------------------------------------------- hardware
            emit("progress", value=0.01, message="probing hardware")
            hw = self.hw or probe(job.dir)
            self._hw = hw
            report["hardware"] = hw.to_dict()
            self._memory_check(report, log, warnings)
            ensure = getattr(self.ctx.runtimes, "ensure_working", None)
            if ensure is not None:          # repair engines whose folder moved (1.x -> 2.0 data folder rename)
                for rid in list(getattr(self.ctx.runtimes, "specs", {})):
                    ensure(rid, log)
            stage("hardware")

            # ---------------------------------------------------- ingest
            for p in job.inputs:
                if not p.exists():
                    raise MediaError(f"Input not found: {p}")
            if len(job.inputs) > 1:
                raise MediaError("One photo or video per job: add the others as their own jobs.")
            kind = classify_path(job.input)
            report["input"] = {"path": str(job.input), "type": kind, "size_bytes": job.input.stat().st_size,
                               "sha256": _sha256(job.input)}
            self._check_disk(hw, job.input)
            inp = JobInput(job.input, kind, job.dir)
            frames_dir = job.dir / "frames"
            if kind == "photo":
                frames_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                img = load_image(job.input)
                ref = frames_dir / "frame_00000.png"
                Image.fromarray(img).save(ref)
                inp.frames.append(ref)
                report["input"]["resolution"] = [int(img.shape[1]), int(img.shape[0])]
            else:
                emit("progress", value=0.03, message="reading the video")
                if not find_ffmpeg():
                    raise MediaError("FFmpeg is missing; cannot read video.")
                info = probe_video(job.input)
                report["input"]["video"] = asdict(info)
                inp.fps = info.fps
            check_cancel()
            stage("ingest")

            # ---------------------------------------------------- select
            sel = select(inp, hw, self.ctx, opts.mode, opts.backend)
            report["selection"] = sel.to_dict()
            if sel.backend is None:
                reasons = "; ".join(f"{r['backend']}: {', '.join(r['reasons'])}" for r in sel.rejected)
                raise BackendError("No backend can process this input on this machine. " + reasons,
                                   code="no_backend")
            backend = sel.backend
            log(f"selected backend: {backend.display_name}")
            warnings += sel.notes

            audio_path = None
            if kind == "video":
                from .backends.video_splat import FRAME_SIDE, video_plan

                info = report["input"]["video"]
                dur = float(info.get("duration_s") or 0) or None
                fps, secs = video_plan(opts.mode, opts.video_fps, opts.video_seconds, info["fps"], dur)
                start = min(max(opts.video_start_s, 0.0), max((dur or secs) - 1.0 / fps, 0.0))
                if dur:
                    secs = min(secs, dur - start)
                emit("progress", value=0.05, message=f"extracting {secs:.1f} s of video at {fps:g} fps")
                inp.frames = extract_frames_fps(job.input, frames_dir, fps, max(1, int(round(secs * fps))),
                                                max_side=FRAME_SIDE, start_s=start, ext="jpg")
                inp.fps = fps
                used = len(inp.frames) / fps
                audio_path = extract_audio(job.input, job.dir / "audio", start, used)
                report["processing"] = {"video_fps": fps, "video_start_s": round(start, 3),
                                        "video_seconds": round(used, 3), "frames": len(inp.frames),
                                        "audio": audio_path is not None}
                if dur and (start > 0.05 or start + used < dur - 0.5):
                    log(f"using {used:.1f} s of the {dur:.1f} s video (from {start:.1f} s)")
                log(f"{len(inp.frames)} frames prepared")
            check_cancel()
            est = backend.estimate(inp, hw, {"mode": opts.mode})
            report["estimate"] = asdict(est)
            stage("select_and_prepare_input")

            # ---------------------------------------------------- reconstruct
            ref_index = 0
            bopts = {"mode": opts.mode, "log": log, "reference_frame_index": ref_index, "fps": inp.fps,
                     "real3d_fill": opts.real3d_fill, "ipd_mm": opts.ipd_mm,
                     "detail": opts.photo_detail or opts.mode == "quality",
                     "audio_path": str(audio_path) if audio_path else None,
                     "video_start_s": report.get("processing", {}).get("video_start_s", 0.0),
                     "hfov_deg": exif_hfov_deg(job.input) if kind == "photo" else None}
            if bopts["hfov_deg"]:
                log(f"EXIF field of view: {bopts['hfov_deg']:.1f}°")
            backend.prepare(self.ctx, bopts)
            emit("progress", value=0.1, message=f"reconstructing with {backend.display_name}")
            result = backend.run(inp, self.ctx, bopts,
                                 lambda v, m: emit("progress", value=0.1 + 0.85 * v, message=m),
                                 lambda: job.cancelled)
            report["backend"] = {"id": backend.id, "name": backend.display_name, "maturity": backend.maturity,
                                 "commercial_use": backend.commercial_use,
                                 "upstream": upstream_commits(backend.upstream),
                                 "runtime": backend.runtime_id,
                                 "worker_env": result.worker_env}
            result.extra.setdefault("hfov_deg", bopts["hfov_deg"])
            result.extra.setdefault("reference_frame_index", ref_index)
            report["backend"]["extra"] = {k: v for k, v in result.extra.items()
                                          if isinstance(v, (str, int, float, bool, dict, list, type(None)))}
            report["models"] = result.models_used
            report["vram_peak_mib"] = result.vram_peak_mib
            warnings += result.warnings
            check_cancel()
            stage("reconstruct")

            # ---------------------------------------------------- export + validate
            emit("progress", value=0.96, message="validating 3D output")
            from .scene import load_scene

            scene = load_scene(result.scene_ply)
            problems = scene.validate()
            if problems:
                warnings += [f"scene: {p}" for p in problems]
                scene = scene.finite_subset()
            if len(scene) == 0:
                raise BackendError("Backend produced an empty scene.", code="empty_output")
            report["coverage"] = {"by_splat": scene.coverage(), "note": result.provenance_note}
            outputs: dict = {"scene_ply": str(result.scene_ply), "splat": str(result.splat) if result.splat else None,
                             "obj": str(result.obj) if result.obj else None, "gaussians": len(scene),
                             "metric_scale": scene.metric_scale, **(result.extra.get("files") or {})}
            if result.extra.get("splv"):
                outputs.update(splv=result.extra["splv"], frames=result.extra.get("frames"),
                               fps=result.extra.get("fps"), splats_per_frame=result.extra.get("splats_per_frame"),
                               audio=result.extra.get("audio"))
            report["validation"] = self._validate_source_view(scene, inp, ref_index)
            stage("export_validate")
            check_cancel()
            report["outputs"] = outputs
            write_export_readme(job.dir / "export", report)
            if opts.output_dir:
                dest = Path(opts.output_dir) / f"{job.input.stem}_{job.id}"
                shutil.copytree(job.dir / "export", dest, dirs_exist_ok=True)
                report["outputs"]["copied_to"] = str(dest)
            report["status"] = "succeeded"
            final_state = "succeeded"
            emit("progress", value=1.0, message="done")
        except JobCancelled as e:
            report["status"] = "cancelled"
            report["error"] = {"code": "cancelled", "message": str(e)}
            final_state = "cancelled"
        except (BackendError, MediaError) as e:
            report["status"] = "failed"
            report["error"] = {"code": getattr(e, "code", "bad_input" if isinstance(e, MediaError) else "error"),
                               "message": str(e)}
            final_state = "failed"
        except Exception as e:  # noqa: BLE001 - job boundary: never crash the app
            report["status"] = "failed"
            report["error"] = {"code": "internal_error", "message": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-4000:]}
            final_state = "failed"
        finally:
            if result is None and "backend" not in report:
                report["backend"] = None
            report["timings_s"] = timings
            report["runtime_s"] = round(sum(timings.values()), 3)
            report["warnings"] = warnings
            report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            report["build"] = {"python": platform.python_version(), "platform": platform.platform()}
            (job.dir / "run_report.json").write_text(json.dumps(report, indent=2, default=str))
            # Only now, with the report on disk, does the job leave the "running" state.
            job.report = report
            job.state = final_state
            try:      # also after a failure or a cancel: a video's frames can take gigabytes
                bid = (report.get("backend") or {}).get("id") or (report.get("selection") or {}).get("backend")
                if bid:
                    from .backends import get_backend

                    get_backend(bid).cleanup(JobInput(job.input, "", job.dir))
            except Exception:
                pass
            log_file.close()
            job.report = report
            emit("finished", status=job.state, report=str(job.dir / "run_report.json"),
                 error=report.get("error"))
        return report

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _memory_check(report: dict, log, warnings: list) -> None:
        """Stop worker processes left over from earlier jobs and say how much memory is left (Windows needs
        RAM + page file for GPU allocations too; when it runs out, models fail in confusing ways)."""
        from .paths import app_paths
        from .sysmem import low_memory_warning, memory_status, stop_stray_workers

        try:
            stray = stop_stray_workers(app_paths().runtimes, log)
        except Exception:  # noqa: BLE001 - a diagnostic, never a reason to fail the job
            stray = []
        st = memory_status()
        if st is not None:
            log("memory before the job: " + st.summary())
            report["hardware"]["memory"] = {"ram_free_gib": round(st.ram_free_gib, 2),
                                            "commit_limit_gib": round(st.commit_limit_gib, 2),
                                            "commit_free_gib": round(st.commit_free_gib, 2),
                                            "stray_workers_stopped": len(stray)}
        w = low_memory_warning(st)
        if w:
            warnings.append(w)
            log("WARNING: " + w)

    def _check_disk(self, hw: HardwareReport, input_path: Path) -> None:
        need_mib = 512 + 20 * input_path.stat().st_size / 2**20
        if hw.disk_free_mib is not None and hw.disk_free_mib < need_mib:
            raise BackendError(f"Insufficient disk space: {hw.disk_free_mib} MiB free, about "
                               f"{need_mib:.0f} MiB needed in {hw.disk_path}.", code="low_disk")

    def _validate_source_view(self, scene, inp: JobInput, ref_index: int = 0, cam=None) -> dict:
        """Render the reference camera and compare with the source frame."""
        from PIL import Image

        from .render import render_pinhole

        if (cam is None and not scene.cameras) or not inp.frames:
            return {"source_view": "skipped (no reference camera)"}
        cam = cam or scene.cameras[0]
        if len(scene) > 1_500_000:          # maximum-detail photos: a sample is enough for this check
            idx = np.sort(np.random.default_rng(0).choice(len(scene), 1_500_000, replace=False))
            sub = scene.subset(idx)
            sub.scales = sub.scales * np.float32(np.sqrt(len(scene) / len(idx)))
            scene = sub
        src = np.asarray(Image.open(inp.frames[min(ref_index, len(inp.frames) - 1)]).convert("RGB"))
        s = min(1.0, 640 / max(cam.width, cam.height))
        w, h = max(8, int(cam.width * s)), max(8, int(cam.height * s))
        src_small = np.asarray(Image.fromarray(src).resize((w, h), Image.BILINEAR)).astype(np.float32)
        r = render_pinhole(scene, cam.c2w, w, h, cam.fx * s, cam.fy * s, cam.cx * s, cam.cy * s)
        cov = r.covered | r.filled
        if cov.sum() == 0:
            return {"source_view": "failed: nothing visible from the reference camera", "psnr_db": None}
        mse = float(np.mean((r.rgb.astype(np.float32)[cov] - src_small[cov]) ** 2))
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))
        return {"source_view": "ok" if psnr > 18 and cov.mean() > 0.8 else "degraded",
                "psnr_db": round(float(psnr), 2), "covered_fraction": round(float(cov.mean()), 4),
                "renderer": "cpu-reference (opaque isotropic splats)", "resolution": [w, h]}
