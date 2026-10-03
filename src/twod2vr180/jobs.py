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
    dynamic_fps: float = 12.0            # per-frame processing rate for fixed-camera dynamic video
    max_dynamic_frames: int = 150
    max_keyframes: int = 80
    license_profile: str = "personal_research"
    output_dir: str | None = None        # copy final outputs here too
    fill_holes: bool = True              # background-fill stereo disocclusions (reported as interpolated)
    renderer: str = "auto"               # auto | gpu | cpu  (VR180 splat renderer)
    allow_cpu: bool = False              # allow CPU inference for cpu-capable backends (slow)
    max_path_frames: int = 720           # cap for moving-camera VR180 videos
    video_mode: str = "auto"             # auto | multiview (all frames → one 3D scene) | per_frame | best_frame
    export_sequence: bool = False        # fixed-camera video: also export one .ply per frame (4D sequence)
    generative: str = "off"              # off | capture | arc | orbit | explore | spiral: invent unseen views
    gen_assembly: str = "fusion"         # fusion (sharp: MoGe-2 per view, merged) | train (one optimised splat)
    gen_engine: str = "auto"             # auto | qwen | wan | seva: model that invents the other views
    generated_inputs: list = field(default_factory=list)   # inputs that are AI views (rebuild with own picks)
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
    lines = [f"2D2VR180 results for {Path((report.get('input') or {}).get('path', '')).name}", "",
             "3D SCENE (explore it in the app's 3D viewer, or with 'View in VR')",
             "  scene.ply    Gaussian splat, standard 3DGS format. Opens in SuperSplat (superspl.at/editor),",
             "               Postshot, Polycam, Luma, Blender (3DGS add-ons) and most splat viewers.",
             "  scene.splat  Same splat in the compact web format (antimatter15 / many web viewers)."]
    if out.get("obj"):
        lines += ["  scene.obj    Triangle mesh with scene.mtl + scene_texture.png (Blender, MeshLab, 3D printing…)."]
    if out.get("sequence_dir") or any(v.get("sequence_dir") for v in vr.get("videos", [])):
        lines += ["  sequence\\    One splat per video frame (frame_00000.ply …): a 4D sequence for players that",
                  "               support splat animations, or to pick a single moment."]
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
                an = analyze_video(job.input, max_keyframes=opts.max_keyframes)
                report["input"]["video"] = an.info
                report["input"]["analysis"] = an.to_dict()
                inp.video_kind = an.kind
                inp.fps = an.info["fps"]
                inp.analysis = an.to_dict()
                for n in an.notes:
                    log(n)
                forced = {"multiview": "moving_camera", "per_frame": "static_camera_dynamic",
                          "best_frame": "static_scene"}.get(opts.video_mode)
                if forced and forced != an.kind:
                    log(f"video mode '{opts.video_mode}' chosen by the user (analysis said '{an.kind}')")
                    inp.video_kind = forced
                    if forced == "moving_camera":
                        # use evenly spaced frames over the whole video
                        total = int(an.info.get("frames") or round((an.info.get("duration_s") or 1) * inp.fps))
                        k = int(min(opts.max_keyframes, max(3, total)))
                        inp.analysis["keyframes"] = [int(x) for x in np.linspace(0, max(total - 1, 0), k).round()]
                    report["input"]["video_mode"] = opts.video_mode
            check_cancel()
            stage("ingest")

            # ---------------------------------------------------- select
            sel = None
            assembly = opts.gen_assembly
            wants_mv = opts.backend == "multiview" and kind == "photo"
            if wants_mv and opts.generative == "off":
                raise BackendError("Multi-view needs several photos of the same scene. For a single photo, choose "
                                   "a generative mode (for example '3 views') to create the other views first.",
                                   code="bad_input")
            if wants_mv:
                assembly = "train"   # generated views + photo → the multi-view engine's trained splat
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
            bopts = {"mode": opts.mode, "log": log, "reference_frame_index": ref_index,
                     "trajectory": opts.generative if opts.generative != "off" else None,
                     "assembly": assembly, "engine": opts.gen_engine,
                     "generated_flags": [str(p) in {str(g) for g in opts.generated_inputs} for p in job.inputs],
                     "hfov_deg": exif_hfov_deg(job.input) if kind == "photo" else None}
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
            if (opts.export_sequence and result.frame_scenes and not (opts.vr180 and find_ffmpeg())):
                for _ in self._frame_scenes(job, result, emit, check_cancel, 0.85, 0.99):
                    pass
                outputs["sequence_dir"] = str(job.dir / "export" / "sequence")
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

    def _stereo_frames(self, job: Job, scene, scene_ply: Path, so, heads: list, emit, lo: float, hi: float,
                       warnings: list, check_cancel, label: str):
        """Yield StereoFrames for the given head poses, on the GPU (gsplat) when
        requested and available, otherwise (or on failure) on the CPU."""
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
        if use_gpu:
            gpu = GpuSplatRenderer(self.ctx, job.dir / "gpu_render")
            done = 0
            try:
                for fr in gpu.frames(scene, scene_ply, so, heads,
                                     lambda v, m: emit("progress", value=lo + (hi - lo) * v, message=f"{label}: {m}"),
                                     lambda: job.cancelled):
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
            yield render_stereo(scene, so, head_c2w=h)

    def _vr180(self, job: Job, scene, inp: JobInput, result: BackendResult, kind: str, emit, warnings,
               check_cancel) -> dict:
        from .vr180 import StereoOptions, save_stereo_image, still_to_video

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
        head = level_c2w(scene.cameras[0].c2w if scene.cameras else np.eye(4), up.up)
        for i, layout in enumerate(opts.layouts):
            check_cancel()
            so = self._stereo_options(opts, layout, opts.eye_resolution)
            lo = 0.75 + 0.05 * i
            fr = next(self._stereo_frames(job, scene, result.scene_ply, so, [head], emit, lo, lo + 0.05,
                                          warnings, check_cancel, f"VR180 {layout.upper()} still"))
            if opts.fill_holes and opts.ai_hole_fill:
                self._ai_fill(job, [fr], emit, lo + 0.04, warnings)
            saved = save_stereo_image(fr, so, vr_dir, job.input.stem)
            out["stills"].append(saved)
            note = fr.metadata.get("honesty_note")
            if note and note not in warnings:
                warnings.append(note)
            if ffmpeg and (inp.kind in ("photo", "images") or kind == "video:static_scene"):
                out["videos"].append(still_to_video(fr, so, vr_dir, job.input.stem, ffmpeg,
                                                    seconds=opts.still_video_seconds))
        if not ffmpeg:
            warnings.append("FFmpeg missing: VR180 video not encoded (stills only).")
            return out
        if kind == "video:static_camera_dynamic" and result.frame_scenes:
            out["videos"].append(self._dynamic_video(job, result, emit, check_cancel, ffmpeg))
        elif kind == "video:moving_camera" and result.cameras:
            out["videos"].append(self._trajectory_video(job, scene, result, inp, emit, check_cancel, ffmpeg,
                                                        warnings))
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

    @staticmethod
    def _depth_stabilisation(result, hfov, ref_index: int = 0) -> list[float]:
        """Per-frame depth scale factors that remove the frame-to-frame scale jitter of monocular
        depth: each frame is scaled to agree with the reference frame on its static pixels
        (where the image has not changed), then the factors are median-filtered over time."""
        from .backends.photo import load_pointmap

        ref_pts, ref_img, _, _ = load_pointmap(result.frame_scenes[ref_index], hfov)
        ref_z = ref_pts[..., 2]
        raw = []
        for npz in result.frame_scenes:
            pts, img, mask, _ = load_pointmap(npz, hfov)
            z = pts[..., 2]
            if z.shape != ref_z.shape:
                raw.append(1.0)
                continue
            static = (np.abs(img.astype(np.int16) - ref_img.astype(np.int16)).max(-1) < 12) & (z > 0) & (ref_z > 0)
            if mask is not None:
                static &= mask.astype(bool)
            raw.append(float(np.median(ref_z[static] / z[static])) if static.sum() > 500 else 1.0)
        raw_a = np.asarray(raw)
        k = 2
        return [float(np.median(raw_a[max(0, i - k):i + k + 1])) for i in range(len(raw_a))]

    def _frame_scenes(self, job, result, emit, check_cancel, lo: float, hi: float):
        """Per-frame splats of a fixed-camera video (depth-stabilised); optionally saved as a
        4D sequence of .ply files in export/sequence."""
        from .backends.photo import load_pointmap
        from .rgbd import pointmap_to_gaussians
        from .scene import write_gaussian_ply

        hfov = result.extra.get("hfov_deg")
        ref = int(result.extra.get("reference_frame_index", 0) or 0)
        scales = self._depth_stabilisation(result, hfov, min(ref, len(result.frame_scenes) - 1))
        n = len(result.frame_scenes)
        seq = job.dir / "export" / "sequence"
        for i, npz in enumerate(result.frame_scenes):
            check_cancel()
            pts, img, mask, cam = load_pointmap(npz, hfov)
            sc = pointmap_to_gaussians(pts * scales[i], img, mask, cam, metric=result.metric_scale)
            if job.options.export_sequence:
                write_gaussian_ply(sc, seq / f"frame_{i:05d}.ply")
            emit("progress", value=lo + (hi - lo) * i / max(n, 1), message=f"video frame {i + 1}/{n}")
            yield sc

    def _dynamic_video(self, job, result, emit, check_cancel, ffmpeg) -> dict:
        from .align import level_c2w
        from .backends.photo import load_pointmap
        from .vr180 import encode_video, output_suffix, render_stereo

        opts = job.options
        so = self._stereo_options(opts, opts.layouts[0], opts.video_eye_resolution)
        hfov = result.extra.get("hfov_deg")
        side = opts.video_eye_resolution
        w, h = (2 * side, side) if so.layout == "sbs" else (side, 2 * side)
        if so.projection == "flat":
            pts, img, mask, cam = load_pointmap(result.frame_scenes[0], hfov)
            w0 = int(round(cam.width * side / cam.height / 2)) * 2
            w, h = (2 * w0, side) if so.layout == "sbs" else (w0, 2 * side)
        up = getattr(self, "_up", None)
        head = level_c2w(np.eye(4), up) if up is not None and so.projection == "equirect180" else None

        def frames():
            # Each frame is a different scene, so the per-frame CPU renderer is used.
            for sc in self._frame_scenes(job, result, emit, check_cancel, 0.85, 0.99):
                yield render_stereo(sc, so, head_c2w=head).image

        meta = encode_video(frames(), w, h, job.options.dynamic_fps, job.dir / "export" / "vr180" /
                            f"{job.input.stem}_dynamic{output_suffix(so)}.mp4", so, ffmpeg)
        meta["method"] = ("per-frame monocular geometry (2.5D), depth scale stabilised over time against the "
                          "static background; moving subjects are reconstructed frame by frame")
        if opts.export_sequence:
            meta["sequence_dir"] = str(job.dir / "export" / "sequence")
        return meta

    def _trajectory_video(self, job, scene, result, inp, emit, check_cancel, ffmpeg, warnings) -> dict:
        from .render import interpolate_poses
        from .scene import Camera
        from .vr180 import encode_video, output_suffix

        opts = job.options
        so = self._stereo_options(opts, opts.layouts[0], opts.video_eye_resolution, "equirect180")
        cams = [Camera.from_dict(c) for c in result.cameras]
        side = opts.video_eye_resolution
        w, h = (2 * side, side) if so.layout == "sbs" else (side, 2 * side)
        dur = (inp.analysis or {}).get("info", {}).get("duration_s") or len(cams) / 2
        fps = 24.0
        n_out = int(max(len(cams), min(opts.max_path_frames, round(dur * fps))))
        heads = interpolate_poses([c.c2w for c in cams], n_out)
        up = getattr(self, "_up", None)
        if up is not None:
            from .align import level_c2w

            heads = [level_c2w(h, up) for h in heads]
        fps = max(1.0, n_out / max(dur, 1e-3))
        frames = (fr.image for fr in self._stereo_frames(job, scene, result.scene_ply, so, heads, emit, 0.85,
                                                          0.99, warnings, check_cancel, "VR180 camera path"))
        meta = encode_video(frames, w, h, fps, job.dir / "export" / "vr180" /
                            f"{job.input.stem}_path{output_suffix(so)}.mp4", so, ffmpeg)
        meta["method"] = ("reconstructed scene rendered along the recovered camera path "
                          f"({len(cams)} keyframe poses interpolated to {n_out} frames)")
        return meta
