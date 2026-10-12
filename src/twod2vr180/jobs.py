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
from .media import (MediaError, analyze_video, classify_path, exif_hfov_deg, extract_frames, extract_frames_fps,
                    find_ffmpeg, load_image)
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
    video_fps: float = 0.0               # 4D video: frames per second used (0 = by quality mode: 8 / 12 / 15)
    video_seconds: float = 0.0           # 4D video: how much of the video (0 = by quality mode: 8 / 12 / 15 s)
    video_start_s: float = 0.0           # 4D video: where to start in the video
    license_profile: str = "personal_research"
    output_dir: str | None = None        # copy final outputs here too
    fill_holes: bool = True              # background-fill stereo disocclusions (reported as interpolated)
    renderer: str = "auto"               # auto | gpu | cpu  (VR180 splat renderer)
    allow_cpu: bool = False              # allow CPU inference for cpu-capable backends (slow)
    generative: str = "off"              # off | capture | arc | orbit | explore | spiral: invent unseen views
    gen_engine: str = "auto"             # auto | qwen | wan: model that invents the other views
    ai_hole_fill: bool = True            # LaMa inpainting of VR180 disocclusions when installed


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
    vr = out.get("vr180") or {}
    lines = [f"2D2VR180 results for {Path((report.get('input') or {}).get('path', '')).name}", ""]
    if out.get("splatv"):
        lines += ["4D SCENE: the video in 3D, over time (explore it in the app's 3D viewer, or with 'View in VR')",
                  "  scene.splatv The whole 4D scene: the still surroundings plus everything that moves, frame by",
                  "               frame. Plays in the app and in splaTV-compatible players (antimatter15.com/splaTV",
                  "               opens a file dragged onto the page; nothing is uploaded).",
                  "  scene.ply    Only the still background (people and other movers left out), standard 3DGS format",
                  "               for SuperSplat, Postshot, Blender and most splat viewers.",
                  "  scene.splat  The background in the compact web format."]
    else:
        lines += ["3D SCENE (explore it in the app's 3D viewer, or with 'View in VR')",
                  "  scene.ply    Gaussian splat, standard 3DGS format. Opens in SuperSplat (superspl.at/editor),",
                  "               Postshot, Polycam, Luma, Blender (3DGS add-ons) and most splat viewers.",
                  "  scene.splat  Same splat in the compact web format (antimatter15 / many web viewers)."]
    if out.get("obj"):
        lines += ["  scene.obj    Triangle mesh with scene.mtl + scene_texture.png (Blender, MeshLab, 3D printing…)."]
    lines += ["  _2d2vr180\\   Metadata the app uses (cameras, which parts are observed/inferred). Keep it next to",
              "               scene.ply if you want to reopen the scene in 2D2VR180; other programs ignore it.", ""]
    if vr.get("stills") or vr.get("videos"):
        lines += ["VR180 (watch in a headset: vr180 folder)",
                  "  *_180_LR.jpg / .mp4   left-right (side-by-side) VR180 stereo",
                  "  *_180_TB.jpg / .mp4   top-bottom VR180 stereo",
                  "  *_coverage.png        white = seen, grey = interpolated, black = unknown",
                  "  Quest: connect by USB, copy the .mp4 into the 'Movies' folder, open it in the Files/Media",
                  "  app or any VR player (DeoVR, Skybox, Pigasus). The files carry VR180 metadata; if a player",
                  "  asks, choose 180° and side-by-side (LR) or top-bottom (TB).", ""]
    lines += ["WALK AROUND THE SPLAT IN VR (6DoF)",
              "  In the app: Results → 'View in VR'. Connect the headset to this PC first (Quest Link,",
              "  Air Link, Virtual Desktop or SteamVR), then press ENTER VR in the page that opens.", "",
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
        final_state = "failed"
        try:
            # ---------------------------------------------------- hardware
            emit("progress", value=0.01, message="probing hardware")
            hw = self.hw or probe(job.dir)
            self._hw = hw
            report["hardware"] = hw.to_dict()
            self._memory_check(report, log, warnings)
            stage("hardware")

            # ---------------------------------------------------- ingest
            for p in job.inputs:
                if not p.exists():
                    raise MediaError(f"Input not found: {p}")
            kind = classify_path(job.input)
            if len(job.inputs) > 1:
                kinds = {classify_path(p) for p in job.inputs}
                if kinds != {"photo"}:
                    raise MediaError("Several inputs can only be combined when they are all photos of the same "
                                     "scene; add videos as separate jobs.")
                kind = "images"
            report["input"] = {"path": str(job.input), "type": kind, "size_bytes": job.input.stat().st_size,
                               "sha256": _sha256(job.input)}
            if kind == "images":
                report["input"]["paths"] = [str(p) for p in job.inputs]
                report["input"]["sha256_all"] = [_sha256(p) for p in job.inputs]
            self._check_disk(hw, job.input)
            inp = JobInput(job.input, kind, job.dir)
            frames_dir = job.dir / "frames"
            if kind in ("photo", "images"):
                frames_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                for i, p in enumerate(job.inputs):
                    img = load_image(p)
                    ref = frames_dir / f"frame_{i:05d}.png"
                    Image.fromarray(img).save(ref)
                    inp.frames.append(ref)
                    if i == 0:
                        report["input"]["resolution"] = [int(img.shape[1]), int(img.shape[0])]
                if kind == "images":
                    log(f"{len(inp.frames)} photos combined into one multi-view scene")
            else:
                emit("progress", value=0.03, message="analysing video")
                if not find_ffmpeg():
                    raise MediaError("FFmpeg is missing; cannot read video.")
                an = analyze_video(job.input)
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
            sel = None
            wants_mv = opts.backend == "multiview" and kind == "photo"
            if wants_mv and opts.generative == "off":
                raise BackendError("Multi-view needs several photos of the same scene. For a single photo, choose "
                                   "a generative mode (for example 'Real 3D') to create the other views first.",
                                   code="bad_input")
            if wants_mv:
                log("multi-view backend on one photo: generating the views first, then a trained multi-view splat")
            if (opts.generative != "off" and (not opts.backend or wants_mv)
                    and (kind == "photo" or inp.video_kind == "static_scene")):
                sel = select(inp, hw, self.ctx, opts.mode, "generative_scene")
                if sel.backend is None:
                    why = "; ".join("; ".join(r["reasons"]) for r in sel.rejected)
                    warnings.append(f"Generative 3D was requested but cannot run: {why}. Used the regular "
                                    "single-photo reconstruction instead.")
                    sel = None
            if sel is None:
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
                src_fps = an_d["info"]["fps"] or 30
                if backend.id == "video4d":      # every frame at a fixed rate → a 4D scene
                    from .backends.video4d import video_plan

                    fps, secs = video_plan(opts.mode, opts.video_fps, opts.video_seconds)
                    fps = min(fps, src_fps)
                    dur = float(an_d["info"].get("duration_s") or secs)
                    start = min(max(opts.video_start_s, 0.0), max(dur - 1.0 / fps, 0.0))
                    inp.frames = extract_frames_fps(job.input, frames_dir, fps, max(2, int(round(secs * fps))),
                                                    max_side=1920, start_s=start)
                    inp.fps = fps
                    used = len(inp.frames) / fps
                    report.setdefault("processing", {}).update(
                        {"video_fps": fps, "video_start_s": round(start, 3), "video_seconds": round(used, 3)})
                    if start + used < dur - 0.5:
                        warnings.append(f"Only {used:.1f} s of the {dur:.1f} s video were used (from {start:.1f} s); "
                                        "change 'Video length' in the options to use more.")
                    # reference frame = the one closest to the sharpest analysed frame
                    k = int(round((an_d["best_frame"] / src_fps - start) * fps))
                    report["processing"]["reference_frame_index"] = min(max(k, 0), len(inp.frames) - 1)
                else:                              # single-frame fallback / Real 3D: the sharpest frame
                    inp.frames = extract_frames(job.input, [an_d["best_frame"]], src_fps, frames_dir)
                log(f"{len(inp.frames)} frame(s) prepared for reconstruction")
            check_cancel()
            est = backend.estimate(inp, hw, {"mode": opts.mode})
            report["estimate"] = asdict(est)
            stage("select_and_prepare_input")

            # ---------------------------------------------------- reconstruct
            ref_index = report.get("processing", {}).get("reference_frame_index", 0)
            bopts = {"mode": opts.mode, "log": log, "reference_frame_index": ref_index, "fps": inp.fps,
                     "trajectory": opts.generative if opts.generative != "off" else None,
                     "engine": opts.gen_engine,
                     "hfov_deg": exif_hfov_deg(job.input) if kind == "photo" else None,
                     # several photos: each one's field of view (the frames are PNG copies without EXIF)
                     "hfov_list": [exif_hfov_deg(p) for p in job.inputs] if kind == "images" else None}
            if bopts["hfov_deg"]:
                log(f"EXIF field of view: {bopts['hfov_deg']:.1f}°")
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
            emit("progress", value=0.72, message="validating 3D output")
            from .scene import load_scene

            scene = load_scene(result.scene_ply)
            problems = scene.validate()
            if problems:
                warnings += [f"scene: {p}" for p in problems]
                scene = scene.finite_subset()
            if len(scene) == 0:
                raise BackendError("Backend produced an empty scene.", code="empty_output")
            s4 = None
            if result.extra.get("scene4d"):
                from .scene4d import Scene4D

                s4 = Scene4D.load(Path(result.extra["scene4d"]))
            report["coverage"] = {"by_splat": scene.coverage(), "note": result.provenance_note}
            outputs: dict = {"scene_ply": str(result.scene_ply), "splat": str(result.splat) if result.splat else None,
                             "obj": str(result.obj) if result.obj else None, "gaussians": len(scene),
                             "metric_scale": scene.metric_scale}
            if s4 is not None:
                outputs.update(scene4d=result.extra["scene4d"], splatv=result.extra.get("splatv"), frames=s4.frames,
                               fps=s4.fps, moving_gaussians=len(s4.dynamic))
            if result.obj is None:
                warnings.append("No OBJ mesh for this backend/input.")
            if s4 is not None:     # the first frame as it was filmed: background + what moved then
                ref_index = min(ref_index, s4.frames - 1)
                report["validation"] = self._validate_source_view(s4.frame(ref_index), inp, ref_index,
                                                                  s4.cameras[ref_index])
            else:
                report["validation"] = self._validate_source_view(scene, inp, ref_index)
            stage("export_validate")
            check_cancel()

            # ---------------------------------------------------- VR180
            if opts.vr180:
                outputs["vr180"] = self._vr180(job, scene, inp, result, sel.effective_kind, emit, warnings,
                                              check_cancel, s4, ref_index)
                stage("vr180")
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

    def _stereo_frames(self, job: Job, scene, scene_ply: Path, so, heads: list, emit, lo: float, hi: float,
                       warnings: list, check_cancel, label: str, s4=None, times: list | None = None):
        """Yield StereoFrames for the given head poses, on the GPU (gsplat) when requested and available,
        otherwise (or on failure) on the CPU. 4D scenes (``s4``): head i shows the scene at ``times[i]``."""
        from .gpu_render import GpuSplatRenderer, gpu_renderer_available
        from .vr180 import render_stereo

        opts = job.options
        n = len(heads)
        use_gpu = False
        if opts.renderer in ("auto", "gpu"):
            ok, why = gpu_renderer_available(self.ctx, self._hw)
            use_gpu = ok
            if not ok and opts.renderer == "gpu":
                warnings.append(f"GPU renderer unavailable ({why}); used the CPU reference renderer.")
        dyn = None
        if s4 is not None:
            times = times or [s4.time_of(0)] * n
            dyn = Path(job.dir / "export" / "_2d2vr180" / "dynamic.npz")
        if use_gpu:
            gpu = GpuSplatRenderer(self.ctx, job.dir / "gpu_render")
            done = 0
            try:
                for fr in gpu.frames(scene, scene_ply, so, heads,
                                     lambda v, m: emit("progress", value=lo + (hi - lo) * v, message=f"{label}: {m}"),
                                     lambda: job.cancelled, dynamic=dyn if dyn is not None and dyn.exists() else None,
                                     times=times):
                    done += 1
                    yield fr
                return
            except JobCancelled:
                raise
            except BackendError as e:
                if done:
                    raise
                warnings.append(f"GPU renderer failed ({e.code}: {str(e)[:200]}); fell back to the CPU renderer.")
        for i, h in enumerate(heads):
            check_cancel()
            emit("progress", value=lo + (hi - lo) * i / max(n, 1), message=f"{label} {i + 1}/{n}")
            yield render_stereo(scene if s4 is None else s4.at(times[i]), so, head_c2w=h)

    def _vr180(self, job: Job, scene, inp: JobInput, result: BackendResult, kind: str, emit, warnings,
               check_cancel, s4=None, ref_index: int = 0) -> dict:
        from .vr180 import save_stereo_image, still_to_video

        opts = job.options
        vr_dir = job.dir / "export" / "vr180"
        out: dict = {"stills": [], "videos": []}
        ffmpeg = find_ffmpeg()
        from .align import estimate_up, level_c2w

        self._lama_note = False
        up = estimate_up(scene)
        out["alignment"] = up.to_dict()
        self._up = up.up
        # Level the virtual head: VR180 with a tilted horizon is uncomfortable to watch.
        if s4 is not None:       # the still image shows the sharpest moment, from where it was filmed
            head = level_c2w(s4.cameras[ref_index].c2w, up.up)
        else:
            head = level_c2w(scene.cameras[0].c2w if scene.cameras else np.eye(4), up.up)
        for i, layout in enumerate(opts.layouts):
            check_cancel()
            so = self._stereo_options(opts, layout, opts.eye_resolution)
            lo = 0.75 + 0.05 * i
            fr = next(self._stereo_frames(job, scene, result.scene_ply, so, [head], emit, lo, lo + 0.05,
                                          warnings, check_cancel, f"VR180 {layout.upper()} still", s4,
                                          [s4.time_of(ref_index)] if s4 is not None else None))
            if opts.fill_holes and opts.ai_hole_fill:
                self._ai_fill(job, [fr], emit, lo + 0.04, warnings)
            saved = save_stereo_image(fr, so, vr_dir, job.input.stem)
            out["stills"].append(saved)
            note = fr.metadata.get("honesty_note")
            if note and note not in warnings:
                warnings.append(note)
            if ffmpeg and s4 is None:
                out["videos"].append(still_to_video(fr, so, vr_dir, job.input.stem, ffmpeg,
                                                    seconds=opts.still_video_seconds))
        if not ffmpeg:
            warnings.append("FFmpeg missing: VR180 video not encoded (stills only).")
            return out
        if s4 is not None and s4.frames > 1:
            out["videos"].append(self._video4d(job, scene, result, s4, emit, check_cancel, ffmpeg, warnings))
        return out

    def _ai_fill(self, job, frames, emit, at: float, warnings: list) -> None:
        from .inpaint import lama_available, lama_fill_frames

        ok, why = lama_available(self.ctx)
        if not ok:
            if not getattr(self, "_lama_note", False):
                warnings.append(f"AI hole filling (LaMa) not used: {why}; holes behind objects were filled by "
                                "stretching the background.")
                self._lama_note = True
            return
        try:
            lama_fill_frames(frames, self.ctx, job.dir / "inpaint", lambda v, m: emit("progress", value=at,
                                                                                     message=m),
                             lambda: job.cancelled)
        except JobCancelled:
            raise
        except BackendError as e:
            warnings.append(f"AI hole filling failed ({e.code}); kept the background fill.")

    @staticmethod
    def _stereo_options(opts: "JobOptions", layout: str, res: int, projection: str | None = None):
        from .vr180 import StereoOptions

        return StereoOptions(layout=layout, projection=projection or opts.projection, eye_resolution=res,
                             eye_separation_m=opts.eye_separation_m, fill_holes=opts.fill_holes)

    VIDEO4D_FPS = 24.0           # smooth 4D scenes are rendered at this rate (they exist between frames)

    def _video4d(self, job, scene, result, s4, emit, check_cancel, ffmpeg, warnings) -> dict:
        """VR180 video of a 4D scene from where it was filmed (camera path smoothed and levelled; a fixed
        camera stays put), the background together with what moved at that moment. A scene whose moving
        parts move smoothly is rendered at 24 fps (camera and time interpolated); a per-frame one at the
        rate its frames were taken."""
        from .align import level_c2w, smooth_c2ws
        from .render import interpolate_poses
        from .vr180 import encode_video, output_suffix

        opts = job.options
        so = self._stereo_options(opts, opts.layouts[0], opts.video_eye_resolution, "equirect180")
        up = getattr(self, "_up", None)
        fixed = s4.meta.get("camera_motion") == "fixed"
        smooth = s4.meta.get("playback") == "continuous"
        c2ws = [c.c2w for c in s4.cameras] or [np.eye(4)] * s4.frames
        path = [c2ws[0]] * s4.frames if fixed else smooth_c2ws(c2ws, sigma=max(1.0, s4.fps / 4))
        fps = max(self.VIDEO4D_FPS, s4.fps) if smooth else s4.fps
        n_out = max(1, int(round(s4.duration_s * fps))) if smooth else s4.frames
        times = [(k + 0.5) / n_out for k in range(n_out)]
        heads = path if n_out == len(path) else interpolate_poses(path, n_out)
        if up is not None:
            heads = [level_c2w(h, up) for h in heads]
        side = opts.video_eye_resolution
        w, h = (2 * side, side) if so.layout == "sbs" else (side, 2 * side)
        frames = (fr.image for fr in self._stereo_frames(job, scene, result.scene_ply, so, heads, emit, 0.85, 0.99,
                                                          warnings, check_cancel, "VR180 4D video", s4, times))
        meta = encode_video(frames, w, h, fps, job.dir / "export" / "vr180" /
                            f"{job.input.stem}_4d{output_suffix(so)}.mp4", so, ffmpeg)
        meta["method"] = ("4D scene: the still background (trained once from all frames) with "
                          + ("the moving parts moving smoothly between frames" if smooth else
                             "the moving parts of each frame")
                          + ", rendered from " + ("the fixed camera position" if fixed else
                                                  "the smoothed, levelled camera path"))
        return meta
