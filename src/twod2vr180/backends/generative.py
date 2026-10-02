"""Photo → generated camera trajectory → real 3D Gaussian splat.

Stable Virtual Camera (Stability AI) imagines the scene from new viewpoints
along a camera path (orbit around the subject, a look-around for VR180, or a
forward-facing spiral); the multi-view engine then reconstructs those views
into one consistent splat. Splats visible from the original photo are labelled
INFERRED, everything else GENERATIVE.
"""

from __future__ import annotations

from .base import (NOVEL_VIEW_COMPLETION, SCENE_STATIC, SINGLE_VIEW, Availability, Backend, BackendContext,
                   BackendResult, Estimate, JobInput, run_worker)
from .multiview import MultiViewBackend

TRAJECTORIES = {
    "arc": "around the subject, ±60° figure-of-eight (most reliable; best for people)",
    "orbit": "full 360° orbit around the main subject (objects; the back side is pure invention)",
    "explore": "look around from where the photo was taken, with a little head movement (best for VR180)",
    "spiral": "small forward-facing spiral (most faithful, least new content)",
}
FRAMES = {"fast": 48, "auto": 80, "quality": 110}


class GenerativeSceneBackend(Backend):
    id = "generative_scene"
    display_name = "Generative 3D from one photo (Stable Virtual Camera + multi-view splats)"
    inputs = ("photo", "video:static_scene")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC, NOVEL_VIEW_COMPLETION})
    runtime_id = "gen-cu128"
    maturity = "experimental"
    upstream = ["stable-virtual-camera", "vggt", "moge", "gsplat"]
    commercial_use = False  # Stable Virtual Camera and VGGT-1B weights are non-commercial
    worker_script = "seva_worker.py"
    description = ("Generates the unseen sides of a photo with a camera-controlled diffusion model and trains a "
                   "real 3D Gaussian splat from the generated views. Invented content is labelled GENERATIVE.")

    def __init__(self):
        self.mv = MultiViewBackend()

    def model_ids(self, options: dict) -> list[str]:
        return ["seva-1.1", "sd21-vae", "clip-vit-h-14"] + self.mv.model_ids(options)

    def min_vram_gb(self, options: dict) -> float | None:
        return 12

    def availability(self, ctx: BackendContext, hw, options: dict | None = None) -> Availability:
        av = super().availability(ctx, hw, options)
        if not ctx.runtimes.is_installed(self.mv.runtime_id):
            av.ok = False
            av.reasons.append(f"runtime '{self.mv.runtime_id}' not installed")
            av.missing_runtime = av.missing_runtime or self.mv.runtime_id
        return av

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        n = FRAMES.get(options.get("mode", "auto"), 80)
        return Estimate(seconds=120 + n * 6 + 400, vram_gb=12, disk_gb=1.0,
                        notes=["estimate; not yet benchmarked on RTX 4080"])

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        traj = options.get("trajectory") or "arc"
        if traj not in TRAJECTORIES:
            traj = "arc"
        n = FRAMES.get(options.get("mode", "auto"), 80)
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        log = options.get("log") or (lambda m: None)
        log(f"generating {n} views along a '{traj}' path: {TRAJECTORIES[traj]}")
        req = {"image": str(ref), "output_dir": str(inp.work_dir / "generated_views"), "trajectory": traj,
               "num_frames": n, "steps": 50, "short_side": 576,
               "hfov_deg": options.get("hfov_deg"), "seed": int(options.get("seed", 23)),
               "seva_dir": str(ctx.models.model_dir("seva-1.1")), "vae_dir": str(ctx.models.model_dir("sd21-vae")),
               "clip_path": str(ctx.models.paths("clip-vit-h-14")["open_clip_model.safetensors"])}
        for m in ("seva-1.1", "sd21-vae"):
            ctx.models.paths(m)  # clear error when missing
        out = run_worker(ctx.runtimes.python(self.runtime_id), self.worker_script, req, inp.work_dir / "worker",
                         progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id), log=options.get("log"),
                         progress_range=(0.0, 0.45), timeout_s=3 * 3600)
        views = out["result"]["views"]
        # The real photo is sampled more often than any single generated view during training.
        w_in = max(4.0, len(views) / 10)
        mv_views = [{"path": v["path"], "generated": bool(v["generated"]), "weight": 1.0 if v["generated"] else w_in}
                    for v in views]
        gen_models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                      for m in ("seva-1.1", "sd21-vae", "clip-vit-h-14")]
        res = self.mv.reconstruct(
            mv_views, inp, ctx, {**options, "max_side": 1024}, progress, cancel, progress_range=(0.45, 0.97),
            extra_models=gen_models,
            extra_warnings=[f"Generative completion: {len(views) - 1} views were invented by Stable Virtual Camera "
                            f"('{traj}' path). Splats seen only in those views are labelled GENERATIVE; they are "
                            "plausible, not measured. Non-commercial licence."])
        res.backend = self.id
        res.extra.update({"trajectory": traj, "generated_views": len(views) - 1,
                          "generation_vram_peak_mib": out["result"].get("vram_peak_mib")})
        res.worker_env = {"generation": out["env"], "reconstruction": res.worker_env}
        return res
