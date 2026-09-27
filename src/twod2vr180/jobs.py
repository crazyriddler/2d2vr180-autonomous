"""Job system: ingest → analyse → select → reconstruct → export → VR180 →
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .backends.base import BackendContext, BackendError, BackendResult, JobCancelled, JobInput
from .hardware import HardwareReport, probe
from .media import (MediaError, analyze_video, classify_path, extract_frames, extract_frames_fps, find_ffmpeg,
                    load_image, probe_video)
from .paths import app_paths, config_dir
from .selector import select

EventFn = Callable[[dict], None]


@dataclass
class JobOptions:
    mode: str = "auto"                   # auto | quality | fast
    backend: str | None = None           # force a backend id
    vr180: bool = True
    layouts: list[str] = field(default_factory=lambda: ["sbs", "tb"])
    projection: str = "equirect180"      # equirect180 | flat
    eye_resolution: int = 2048
    video_eye_resolution: int = 1280
    eye_separation_m: float = 0.064
    still_video_seconds: float = 5.0     # length of the VR180 MP4 made from a still scene
    dynamic_fps: float = 12.0            # per-frame processing rate for fixed-camera dynamic video
    max_dynamic_frames: int = 150
    max_keyframes: int = 80
    license_profile: str = "personal_research"
    output_dir: str | None = None        # copy final outputs here too


class Job:
    def __init__(self, input_path: Path, options: JobOptions | None = None, jobs_root: Path | None = None):
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.input = Path(input_path)
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


class JobRunner:
    def __init__(self, ctx: BackendContext, hardware: HardwareReport | None = None):
        self.ctx = ctx
        self.hw = hardware

    # ------------------------------------------------------------------ run
    def run(self, job: Job, on_event: EventFn | None = None) -> dict:
        job.dir.mkdir(parents=True, exist_ok=True)
        opts = job.options
        self.ctx.license_profile = opts.license_profile
        timings: dict[str, float] = {}
        warnings: list[str] = []
        logs: list[str] = []
        report: dict = {
            "schema": "2d2vr180.run_report/1",
            "app": {"name": "2D2VR180", "version": __version__},
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
        try:
            # ---------------------------------------------------- hardware
            emit("progress", value=0.01, message="probing hardware")
            hw = self.hw or probe(job.dir)
            report["hardware"] = hw.to_dict()
            stage("hardware")

            # ---------------------------------------------------- ingest
            if not job.input.exists():
                raise MediaError(f"Input not found: {job.input}")
            kind = classify_path(job.input)
            report["input"] = {"path": str(job.input), "type": kind, "size_bytes": job.input.stat().st_size,
                               "sha256": _sha256(job.input)}
            self._check_disk(hw, job.input)
            inp = JobInput(job.input, kind, job.dir)
            frames_dir = job.dir / "frames"
            if kind == "photo":
                img = load_image(job.input)
                frames_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                ref = frames_dir / "frame_00000.png"
                Image.fromarray(img).save(ref)
                inp.frames = [ref]
                report["input"]["resolution"] = [int(img.shape[1]), int(img.shape[0])]
            else:
                emit("progress", value=0.03, message="analysing video")
                if not find_ffmpeg():
                    raise MediaError("FFmpeg is missing; cannot read video.")
                an = analyze_video(job.input, max_keyframes=opts.max_keyframes)
                report["input"]["video"] = an.info
                report["input"]["analysis"] = an.to_dict()
                inp.video_kind = an.kind
                inp.fps = an.info["fps"]
                inp.analysis = an.to_dict()
                for n in an.notes:
                    log(n)
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
            log(f"selected backend: {backend.display_name} ({sel.effective_kind})")
            for n in sel.notes:
                warnings.append(n)

            if kind == "video":
                emit("progress", value=0.06, message="extracting frames")
                an_d = inp.analysis or {}
                if sel.effective_kind == "video:static_camera_dynamic":
                    fps = min(opts.dynamic_fps, inp.fps or opts.dynamic_fps)
                    inp.frames = extract_frames_fps(job.input, frames_dir, fps, opts.max_dynamic_frames)
                    inp.fps = fps
                    # reference frame = the one closest to the sharpest analysed frame
                    ref_t = an_d["best_frame"] / (an_d["info"]["fps"] or 30)
                    report.setdefault("processing", {})["reference_frame_index"] = min(
                        int(round(ref_t * fps)), len(inp.frames) - 1)
                elif sel.effective_kind == "video:static_scene":
                    inp.frames = extract_frames(job.input, [an_d["best_frame"]], an_d["info"]["fps"], frames_dir)
                else:
                    inp.frames = extract_frames(job.input, an_d["keyframes"], an_d["info"]["fps"], frames_dir,
                                                max_side=1920)
                log(f"{len(inp.frames)} frame(s) prepared for reconstruction")
            check_cancel()
            est = backend.estimate(inp, hw, {"mode": opts.mode})
            report["estimate"] = asdict(est)
            stage("select_and_prepare_input")

            # ---------------------------------------------------- reconstruct
            ref_index = report.get("processing", {}).get("reference_frame_index", 0)
            bopts = {"mode": opts.mode, "log": log, "reference_frame_index": ref_index}
            backend.prepare(self.ctx, bopts)
            emit("progress", value=0.1, message=f"reconstructing with {backend.display_name}")
            result = backend.run(inp, self.ctx, bopts,
                                 lambda v, m: emit("progress", value=0.1 + 0.6 * v, message=m),
                                 lambda: job.cancelled)
            report["backend"] = {"id": backend.id, "name": backend.display_name, "maturity": backend.maturity,
                                 "commercial_use": backend.commercial_use,
                                 "upstream": upstream_commits(backend.upstream),
                                 "runtime": backend.runtime_id,
                                 "worker_env": result.worker_env}
            report["models"] = result.models_used
            report["vram_peak_mib"] = result.vram_peak_mib
            warnings += result.warnings
            check_cancel()
            stage("reconstruct")

            # ---------------------------------------------------- export + validate
            emit("progress", value=0.72, message="validating 3D output")
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
                             "metric_scale": scene.metric_scale}
            if result.obj is None:
                warnings.append("No OBJ mesh for this backend/input.")
            report["validation"] = self._validate_source_view(scene, inp, ref_index)
            stage("export_validate")
            check_cancel()

            # ---------------------------------------------------- VR180
            if opts.vr180:
                outputs["vr180"] = self._vr180(job, scene, inp, result, sel.effective_kind, emit, warnings,
                                              check_cancel)
                stage("vr180")
            report["outputs"] = outputs
            if opts.output_dir:
                dest = Path(opts.output_dir) / f"{job.input.stem}_{job.id}"
                shutil.copytree(job.dir / "export", dest, dirs_exist_ok=True)
                report["outputs"]["copied_to"] = str(dest)
            report["status"] = "succeeded"
            job.state = "succeeded"
            emit("progress", value=1.0, message="done")
        except JobCancelled as e:
            report["status"] = "cancelled"
            report["error"] = {"code": "cancelled", "message": str(e)}
            job.state = "cancelled"
        except (BackendError, MediaError) as e:
            report["status"] = "failed"
            report["error"] = {"code": getattr(e, "code", "bad_input" if isinstance(e, MediaError) else "error"),
                               "message": str(e)}
            job.state = "failed"
        except Exception as e:  # noqa: BLE001 - job boundary: never crash the app
            report["status"] = "failed"
            report["error"] = {"code": "internal_error", "message": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-4000:]}
            job.state = "failed"
        finally:
            if result is None and "backend" not in report:
                report["backend"] = None
            report["timings_s"] = timings
            report["runtime_s"] = round(sum(timings.values()), 3)
            report["warnings"] = warnings
            report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            report["build"] = {"python": platform.python_version(), "platform": platform.platform()}
            (job.dir / "run_report.json").write_text(json.dumps(report, indent=2, default=str))
            try:
                if job.state == "succeeded":
                    from .backends import get_backend

                    get_backend(report["backend"]["id"]).cleanup(JobInput(job.input, "", job.dir))
            except Exception:
                pass
            log_file.close()
            job.report = report
            emit("finished", status=job.state, report=str(job.dir / "run_report.json"),
                 error=report.get("error"))
        return report

    # ------------------------------------------------------------------ helpers
    def _check_disk(self, hw: HardwareReport, input_path: Path) -> None:
        need_mib = 512 + 20 * input_path.stat().st_size / 2**20
        if hw.disk_free_mib is not None and hw.disk_free_mib < need_mib:
            raise BackendError(f"Insufficient disk space: {hw.disk_free_mib} MiB free, about "
                               f"{need_mib:.0f} MiB needed in {hw.disk_path}.", code="low_disk")

    def _validate_source_view(self, scene, inp: JobInput, ref_index: int = 0) -> dict:
        """Render the reference camera and compare with the source frame."""
        from PIL import Image

        from .render import render_pinhole

        if not scene.cameras or not inp.frames:
            return {"source_view": "skipped (no reference camera)"}
        cam = scene.cameras[0]
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

    def _vr180(self, job: Job, scene, inp: JobInput, result: BackendResult, kind: str, emit, warnings,
               check_cancel) -> dict:
        from .vr180 import StereoOptions, encode_video, render_stereo, save_stereo_image, still_to_video

        opts = job.options
        vr_dir = job.dir / "export" / "vr180"
        out: dict = {"stills": [], "videos": []}
        ffmpeg = find_ffmpeg()
        head = scene.cameras[0].c2w if scene.cameras else np.eye(4)
        for i, layout in enumerate(opts.layouts):
            check_cancel()
            emit("progress", value=0.75 + 0.05 * i, message=f"VR180 {layout.upper()} still")
            so = StereoOptions(layout=layout, projection=opts.projection, eye_resolution=opts.eye_resolution,
                               eye_separation_m=opts.eye_separation_m)
            fr = render_stereo(scene, so, head_c2w=head)
            saved = save_stereo_image(fr, so, vr_dir, job.input.stem)
            out["stills"].append(saved)
            note = fr.metadata.get("honesty_note")
            if note and note not in warnings:
                warnings.append(note)
            if ffmpeg and (inp.kind == "photo" or kind == "video:static_scene"):
                out["videos"].append(still_to_video(fr, so, vr_dir, job.input.stem, ffmpeg,
                                                    seconds=opts.still_video_seconds))
        if not ffmpeg:
            warnings.append("FFmpeg missing: VR180 video not encoded (stills only).")
            return out
        if kind == "video:static_camera_dynamic" and result.frame_scenes:
            out["videos"].append(self._dynamic_video(job, result, emit, check_cancel, ffmpeg))
        elif kind == "video:moving_camera" and result.cameras:
            out["videos"].append(self._trajectory_video(job, scene, result, inp, emit, check_cancel, ffmpeg))
        return out

    def _dynamic_video(self, job, result, emit, check_cancel, ffmpeg) -> dict:
        from .backends.photo import load_pointmap
        from .rgbd import pointmap_to_gaussians
        from .vr180 import StereoOptions, encode_video, render_stereo

        opts = job.options
        so = StereoOptions(layout=opts.layouts[0], projection=opts.projection,
                           eye_resolution=opts.video_eye_resolution, eye_separation_m=opts.eye_separation_m)
        n = len(result.frame_scenes)
        side = opts.video_eye_resolution
        w, h = (2 * side, side) if so.layout == "sbs" else (side, 2 * side)
        if so.projection == "flat":
            pts, img, mask, cam = load_pointmap(result.frame_scenes[0])
            w0 = int(round(cam.width * side / cam.height / 2)) * 2
            w, h = (2 * w0, side) if so.layout == "sbs" else (w0, 2 * side)

        def frames():
            for i, npz in enumerate(result.frame_scenes):
                check_cancel()
                pts, img, mask, cam = load_pointmap(npz)
                sc = pointmap_to_gaussians(pts, img, mask, cam)
                emit("progress", value=0.85 + 0.14 * i / n, message=f"VR180 video frame {i + 1}/{n}")
                yield render_stereo(sc, so).image

        from .vr180 import output_suffix

        meta = encode_video(frames(), w, h, job.options.dynamic_fps, job.dir / "export" / "vr180" /
                            f"{job.input.stem}_dynamic{output_suffix(so)}.mp4", so, ffmpeg)
        meta["method"] = ("per-frame monocular geometry (2.5D): each frame is reconstructed independently; "
                          "depth may flicker between frames")
        return meta

    def _trajectory_video(self, job, scene, result, inp, emit, check_cancel, ffmpeg) -> dict:
        from .scene import Camera
        from .vr180 import StereoOptions, encode_video, output_suffix, render_stereo

        opts = job.options
        so = StereoOptions(layout=opts.layouts[0], projection="equirect180",
                           eye_resolution=opts.video_eye_resolution, eye_separation_m=opts.eye_separation_m)
        cams = [Camera.from_dict(c) for c in result.cameras]
        side = opts.video_eye_resolution
        w, h = (2 * side, side) if so.layout == "sbs" else (side, 2 * side)
        dur = (inp.analysis or {}).get("info", {}).get("duration_s") or len(cams)
        fps = max(1.0, min(30.0, len(cams) / max(dur, 1e-3)))

        def frames():
            for i, c in enumerate(cams):
                check_cancel()
                emit("progress", value=0.85 + 0.14 * i / len(cams), message=f"VR180 path frame {i + 1}/{len(cams)}")
                yield render_stereo(scene, so, head_c2w=c.c2w).image

        meta = encode_video(frames(), w, h, fps, job.dir / "export" / "vr180" /
                            f"{job.input.stem}_path{output_suffix(so)}.mp4", so, ffmpeg)
        meta["method"] = "reconstructed scene rendered along the recovered camera path (keyframe poses)"
        return meta
