"""Splat video (2D2VR 2.0): every frame of a video made with the Real 3D technique, played back as a video.

Each frame is what Real 3D makes of a photo, without the AI fill (too slow for thousands of frames): MoGe-3's
measured depth at the video's lens, the people's shape from Sapiens2, the real-world scale from the eyes, and
Apple SHARP aligned to that depth (workers/video_splat_worker.py). The lens and the scale are measured once
for the whole clip; each frame's depth is matched to the previous one where the picture did not change.

The frames and the video's sound go into one file, scene.splv (workers/splv_codec.py), which the app's 3D
viewer and the browser VR player stream and play at the video's speed. The first frame is also exported as
scene.ply / scene.splat for other splat programs.

Memory: the worker loads every model once and writes each frame to disk as soon as it is made, so RAM and
VRAM stay flat however long the video is. When the GPU runs out of memory anyway, the run resumes from the
last finished frame with lighter settings (people's model off, then a smaller depth map).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..scene import INFERRED, write_gaussian_ply, write_splat
from .base import DYNAMIC, SCENE_STATIC, Backend, BackendContext, BackendError, BackendResult, Estimate, JobInput, run_worker
from .locked import FACE_MODEL, HUMAN_POINTMAP, HUMAN_SEG

MOGE = "moge-3-vitl"
# per quality mode: SHARP grid steps (front, back layer: splats per frame), depth map size, seconds per frame
PLANS = {"fast": {"grid": [2.0, 4.0], "moge_max_side": 768, "s_per_frame": 2.5},
         "auto": {"grid": [1.5, 3.0], "moge_max_side": 1024, "s_per_frame": 3.5},
         "quality": {"grid": [1.25, 2.0], "moge_max_side": 1536, "s_per_frame": 5.0}}
FRAME_SIDE = 1536            # frames are extracted at most this big (SHARP's own input size)
BYTES_PER_SPLAT = 10         # after compression, a keyframe of a real video (estimate for the disk check)


def splats_per_frame(grid) -> int:
    import numpy as np

    n = 0
    for s in grid:
        n += len(np.unique(np.minimum(np.round(np.arange(0, 768, max(float(s), 1.0))), 767))) ** 2
    return n


def video_plan(mode: str, fps: float, seconds: float, src_fps: float, duration: float | None) -> tuple[float, float]:
    """Frames per second and seconds of video to make: the video's own rate up to 24 fps and the whole
    video unless the options say otherwise."""
    fps = float(fps) if fps and fps > 0 else min(float(src_fps or 24.0), 24.0)
    fps = min(fps, float(src_fps or fps))
    secs = float(seconds) if seconds and seconds > 0 else float(duration or 10.0)
    return fps, secs


class SplatVideoBackend(Backend):
    id = "splat_video"
    display_name = "Splat video — Real 3D on every frame (Apple SHARP + measured depth)"
    inputs = ("video",)
    capabilities = frozenset({SCENE_STATIC, DYNAMIC})
    runtime_id = "photo-cu128"
    maturity = "experimental"
    upstream = ["sharp", "moge", "sapiens2"]
    commercial_use = False          # SHARP weights: research only
    worker_script = "video_splat_worker.py"
    description = ("A video → a splat video: every frame made like a Real 3D photo (SHARP aligned to MoGe-3's "
                   "measured depth, people's shape from Sapiens2, scale from the eyes), with the sound.")

    def model_ids(self, options: dict) -> list[str]:
        return ["sharp", MOGE]

    def min_vram_gb(self, options: dict) -> float | None:
        return 8

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        plan = PLANS.get(options.get("mode", "auto"), PLANS["auto"])
        n = len(inp.frames) or 1
        size = n * splats_per_frame(plan["grid"]) * BYTES_PER_SPLAT / 2**30
        return Estimate(seconds=30 + n * plan["s_per_frame"], vram_gb=10, disk_gb=round(2.2 * size, 2),
                        notes=["estimate; about " + f"{plan['s_per_frame']:.1f} s per frame on an RTX 4080"])

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        from .. import splv

        log = options.get("log") or (lambda m: None)
        mode = options.get("mode", "auto")
        plan = PLANS.get(mode, PLANS["auto"])
        if not inp.frames:
            raise BackendError("no video frames to process", code="bad_input")
        work = inp.work_dir / "splat_video"
        work.mkdir(parents=True, exist_ok=True)
        exp = inp.work_dir / "export"
        exp.mkdir(parents=True, exist_ok=True)
        fps = float(options.get("fps") or inp.fps or 24.0)
        n = len(inp.frames)
        count = splats_per_frame(plan["grid"])
        need = 2.2 * n * count * BYTES_PER_SPLAT
        free = shutil.disk_usage(work).free
        if free < need:
            raise BackendError(f"Not enough disk space for this splat video: about {need / 2**30:.1f} GB needed, "
                               f"{free / 2**30:.1f} GB free. Use a shorter part of the video, fewer frames per "
                               "second or the Fast mode.", code="low_disk")
        log(f"splat video: {n} frames at {fps:g} fps, {count:,} splats per frame; about "
            f"{n * plan['s_per_frame'] / 60:.0f} min on an RTX 4080, {n * count * BYTES_PER_SPLAT / 2**30:.1f} GB")
        models_used = []

        def used(mid):
            st = ctx.models.status(mid)
            models_used.append({"id": mid, **{k: st[k] for k in ("revision", "license", "hash_status")}})

        def installed_file(mid):
            if mid in ctx.models.entries and ctx.models.is_installed(mid):
                used(mid)
                return str(next(iter(ctx.models.paths(mid).values())))
            return None

        face = installed_file(FACE_MODEL)
        human_seg, human_pm = installed_file(HUMAN_SEG), installed_file(HUMAN_POINTMAP)
        warnings = []
        if not (human_seg and human_pm):
            human_seg = human_pm = None
            log("people's own 3D shape needs Sapiens2 (Components); the scene's depth is used for them")
        req = {"frames": [str(p) for p in inp.frames], "output_dir": str(work),
               "checkpoint": str(ctx.models.paths("sharp")["sharp_2572gikvuh.pt"]),
               "moge_path": str(ctx.models.paths(MOGE)["model.pt"]), "face_model": face, "human_seg": human_seg,
               "human_pointmap": human_pm, "ipd_target_m": float(options.get("ipd_mm") or 63.0) / 1000.0,
               "fov_x": options.get("hfov_deg"), "grid": plan["grid"], "keyframe_interval": max(1, round(fps)),
               "moge_max_side": plan["moge_max_side"], "people": True,
               "allow_cpu": bool(getattr(ctx, "allow_cpu", False))}
        py, env = ctx.runtimes.python(self.runtime_id), ctx.runtimes.worker_env(self.runtime_id)
        lighter = [("the people's model off", {"people": False}),
                   ("a smaller depth map", {"people": False, "moge_max_side": 768})]
        while True:
            try:
                out = run_worker(py, self.worker_script, req, inp.work_dir / "worker", progress, cancel, env=env,
                                 log=log, progress_range=(0.0, 0.92))
                break
            except BackendError as e:
                if e.code != "oom" or not lighter:
                    raise
                what, change = lighter.pop(0)
                req.update(change)
                warnings.append(f"The GPU ran out of memory; the rest of the video was made with {what}.")
                log(warnings[-1] + " Resuming from the last finished frame.")
        used("sharp")
        used(MOGE)
        res = out["result"]

        # one file: the frames and the sound
        progress(0.93, "writing the splat video")
        fdir = Path(res["outputs"]["frames_dir"])
        audio, mime = None, None
        if options.get("audio_path") and Path(options["audio_path"]).exists():
            from ..media import AUDIO_MIME

            audio = Path(options["audio_path"]).read_bytes()
            mime = AUDIO_MIME.get(Path(options["audio_path"]).suffix.lower(), "audio/mp4")
        meta = json.loads(Path(res["outputs"]["meta"]).read_text())

        def chunks():
            for i in range(n):
                p = fdir / f"{i:06d}.bin"
                key = json.loads((fdir / f"{i:06d}.bin.json").read_text())["key"]
                yield p.read_bytes(), key

        header = {"fps": fps, "count": int(res["count"]), "keyframe_interval": req["keyframe_interval"],
                  "quant": res["quant"], "camera": res["camera"],
                  "app": "2D2VR", "source": Path(inp.path).name, "scale": res.get("scale"),
                  "scale_from": res.get("scale_from"), "grid": meta.get("grid"),
                  "start_s": options.get("video_start_s", 0.0)}
        video = exp / "scene.splv"
        head = splv.codec().write_file(video, header, chunks(), audio=audio, audio_mime=mime or "audio/mp4")
        reader = splv.open_reader(video)
        first = splv.frame_scene(reader, 0)
        first.provenance[:] = INFERRED
        ply = write_gaussian_ply(first, exp / "scene.ply", include_provenance=True)
        splat = write_splat(first, exp / "scene.splat")
        shutil.rmtree(fdir, ignore_errors=True)       # the frames now live in scene.splv
        progress(1.0, "done")
        if audio is None:
            warnings.append("The video has no sound (or it could not be read); the splat video is silent.")
        if res.get("scale_from") == "eye distance":
            warnings.append(f"Real-world scale set from the eyes of the people in the video: ×{res['scale']:.3f}.")
        elif res.get("scale_from") == "human model":
            warnings.append(f"Real-world scale set from the human model's distance of the people: "
                            f"×{res['scale']:.3f}.")
        warnings.append("OBJ mesh is not available for splat videos.")
        size = video.stat().st_size
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=None, cameras=[res["camera"]], metric_scale=True,
            provenance_note=(f"{head['count']:,} splats per frame, every one INFERRED from its frame: Apple SHARP "
                             "aligned to the measured depth (MoGe-3). Nothing hidden in a frame is invented."),
            vram_peak_mib=res.get("vram_peak_mib"), worker_env=out["env"], models_used=models_used,
            warnings=warnings,
            extra={"splv": str(video), "frames": n, "fps": fps, "splats_per_frame": int(head["count"]),
                   "size_mb": round(size / 2**20, 1), "audio": audio is not None, "scale": res.get("scale"),
                   "scale_from": res.get("scale_from"), "fov_x": res.get("fov_x"),
                   "ipd_target_mm": round(req["ipd_target_m"] * 1000, 1), "grid": plan["grid"],
                   "files": {"splv": str(video)}})

    def cleanup(self, inp: JobInput) -> None:
        super().cleanup(inp)
        shutil.rmtree(inp.work_dir / "frames", ignore_errors=True)     # extracted frames: in scene.splv now
        shutil.rmtree(inp.work_dir / "splat_video" / "frames", ignore_errors=True)   # left by a failed run
        for p in inp.work_dir.glob("audio.*"):
            p.unlink(missing_ok=True)
