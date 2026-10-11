"""4D video: a video becomes a 4D Gaussian-splat scene (3D that changes over time).

VGGT poses every frame → MoGe-2 metric depth of every frame, aligned to the joint geometry → what
moves is found by reprojecting frames into each other → the still part is trained once, as one 3D
Gaussian splat, from all frames with the moving pixels left out → the moving parts become per-frame
splats in the same world, each visible only at its own time. Runs in 'recon3d-cu124'
(workers/video4d_worker.py). Output: scene.ply (the background, for any splat viewer), scene.splatv
(the whole 4D scene, splaTV format) and the full-precision 4D scene for the app.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..scene import Camera, read_gaussian_ply, write_splat
from ..scene4d import DynamicSplats, Scene4D
from .base import (DYNAMIC, MULTI_VIEW, SCENE_STATIC, Backend, BackendContext, BackendResult, Estimate, JobInput,
                   run_worker)

# mode → (frames per second, seconds of video, training steps, longest image side, moving splats per
#         frame, background splat cap)
QUALITY = {
    "fast": (8.0, 8.0, 5000, 720, 25_000, 1_000_000),
    "auto": (12.0, 12.0, 9000, 960, 40_000, 1_500_000),
    "quality": (15.0, 15.0, 14000, 1280, 60_000, 2_000_000),
}
MOTION_STEPS = {"fast": 2000, "auto": 4000, "quality": 6000}   # smooth 4D motion of the moving parts


def video_plan(mode: str, fps: float | None = None, seconds: float | None = None) -> tuple[float, float]:
    """(frames per second, seconds) used for a 4D video: the user's choice, else the mode's."""
    q = QUALITY.get(mode, QUALITY["auto"])
    return float(fps or q[0]), float(seconds or q[1])


class Video4DBackend(Backend):
    id = "video4d"
    display_name = "4D video (VGGT + MoGe-2 + gsplat)"
    inputs = ("video:moving_camera", "video:static_camera_dynamic", "video:static_scene")
    capabilities = frozenset({MULTI_VIEW, SCENE_STATIC, DYNAMIC})
    runtime_id = "recon3d-cu124"
    maturity = "experimental"
    upstream = ["vggt", "moge", "gsplat"]
    commercial_use = False  # facebook/VGGT-1B checkpoint is non-commercial
    worker_script = "video4d_worker.py"
    description = ("A video becomes a 4D scene: the still surroundings are trained once from all frames (what "
                   "someone hides in one frame is filled in from the others), people, animals and anything else "
                   "that moves are rebuilt frame by frame in the same 3D space. Plays in the app's 3D viewer and "
                   "in VR; the VR180 video is rendered from it.")

    def model_ids(self, options: dict) -> list[str]:
        return ["vggt-1b", "moge-2-vitl-normal"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 10

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        fps, secs, steps, side, _, _ = QUALITY.get(options.get("mode", "auto"), QUALITY["auto"])
        n = max(len(inp.frames), int(fps * secs))
        return Estimate(seconds=60 + n * 2.0 + steps * 0.035 * (side / 960) ** 2, vram_gb=12,
                        disk_gb=0.5 + 0.02 * n, notes=["estimate; not yet benchmarked on RTX 4080"])

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        mode = options.get("mode", "auto")
        _, _, steps, side, fg_budget, cap = QUALITY.get(mode, QUALITY["auto"])
        fps = float(options.get("fps") or inp.fps or 12.0)
        out_dir = inp.work_dir / "video4d_out"
        # the analysis saw the whole image shift: trust the per-frame poses; otherwise the worker decides
        hint = "moving" if inp.video_kind == "moving_camera" else "auto"
        req = {"frames": [str(p) for p in inp.frames], "fps": fps, "output_dir": str(out_dir),
               "vggt_dir": str(ctx.models.model_dir("vggt-1b")),
               "moge_path": str(ctx.models.paths("moge-2-vitl-normal")["model.pt"]),
               "max_side": int(options.get("max_side", side)), "steps": int(options.get("train_steps", steps)),
               "max_gaussians": cap, "fg_budget": fg_budget, "camera": hint, "mesh": bool(options.get("mesh", True)),
               "motion": bool(options.get("motion", True)), "motion_steps": MOTION_STEPS.get(mode, 4000)}
        ctx.models.paths("vggt-1b")  # clear error when missing
        out = run_worker(ctx.runtimes.python(self.runtime_id), self.worker_script, req, inp.work_dir / "worker",
                         progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id), log=options.get("log"),
                         progress_range=(0.0, 0.92), timeout_s=6 * 3600)
        res = out["result"]
        files = res["outputs"]
        progress(0.93, "saving the 4D scene")
        cams = [Camera.from_dict(c) for c in json.loads(Path(files["cameras"]).read_text())]
        bg = read_gaussian_ply(Path(files["background"]))
        bg.cameras = cams[:1]
        bg.metric_scale = bool(res.get("metric"))
        smooth = res.get("moving_model") == "motion"
        s4 = Scene4D(bg, DynamicSplats.load(Path(files["dynamic"])), float(res.get("fps", fps)),
                     int(res.get("frames", len(cams))), cams,
                     {"playback": "continuous" if smooth else "frames", "camera_motion": res.get("camera_motion"),
                      "moving_model": res.get("moving_model", "per_frame"), "source": inp.path.name})
        exp = inp.work_dir / "export"
        saved = s4.save(exp)
        splat = write_splat(bg, exp / "scene.splat")
        obj = None
        if files.get("obj") and Path(files["obj"]).exists():
            obj = exp / "scene.obj"
            shutil.copy2(files["obj"], obj)
        preview = None
        if files.get("moving_preview") and Path(files["moving_preview"]).exists():
            preview = exp / "_2d2vr180" / "moving_preview.png"
            shutil.copy2(files["moving_preview"], preview)
        p = res.get("provenance") or {}
        moving = res.get("moving_fraction") or []
        fixed = res.get("camera_motion") == "fixed"
        note = (f"4D scene from {s4.frames} frames at {s4.fps:g} fps. Background: "
                + ("one viewpoint (fixed camera), its depth is INFERRED; what moving people/objects hid in one "
                   "frame was filled in from other frames" if fixed else
                   f"splats seen from two or more camera positions are OBSERVED ({p.get('observed', 0):,})")
                + f". Moving parts ({res.get('moving_splats', 0):,} splats): INFERRED, the side the camera saw only; "
                + ("Gaussians that move smoothly between frames." if smooth else "rebuilt frame by frame."))
        warnings = []
        if not res.get("metric"):
            warnings.append("Metric scale could not be estimated; stereo depth may need the eye-separation setting.")
        if moving and max(moving) > 0.6:
            warnings.append(f"Up to {max(moving):.0%} of a frame was found to move: little still background was "
                            "visible in those frames.")
        models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                  for m in self.model_ids(options)]
        progress(1.0, "done")
        return BackendResult(
            backend=self.id, scene_ply=Path(saved["scene_ply"]), splat=splat, obj=obj,
            cameras=[c.to_dict() for c in cams], metric_scale=bg.metric_scale, provenance_note=note,
            vram_peak_mib=res.get("vram_peak_mib"), worker_env=out["env"], models_used=models, warnings=warnings,
            extra={"scene4d": saved["scene4d"], "splatv": saved.get("splatv"), "frames": s4.frames, "fps": s4.fps,
                   "moving_model": s4.meta["moving_model"],
                   "duration_s": round(s4.duration_s, 3), "camera_motion": res.get("camera_motion"),
                   "moving_fraction_mean": round(sum(moving) / len(moving), 4) if moving else 0.0,
                   "moving_fraction_max": round(max(moving), 4) if moving else 0.0,
                   "background_splats": res.get("background_splats"), "moving_splats": res.get("moving_splats"),
                   "reference_psnr_db": res.get("reference_psnr_db"), "mesh": res.get("mesh"),
                   "moving_preview": str(preview) if preview else None,
                   "metric_scale_factor": res.get("metric_scale_factor")})
