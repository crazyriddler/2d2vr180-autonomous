"""Photo → generated camera moves → real 3D Gaussian splat.

Two generation engines, chosen automatically:
  * Wan 2.2 Fun 5B Control-Camera (Alibaba PAI, Apache-2.0) - a video model trained on real footage:
    camera-controlled shots that start at the photo; keeps people and animals intact (preferred);
  * Stable Virtual Camera (Stability AI, non-commercial) - multi-view diffusion; good for scenes and
    objects, documented to distort people.
The views are then assembled into one 3D scene, by default with *sharp fusion*: VGGT poses the
views, MoGe-2 gives each one crisp per-pixel geometry, and only surfaces the photo does not show are
added from the generated views (no optimisation, so nothing is blurred). The alternative, *trained*,
optimises one splat against every generated frame. Photo pixels are INFERRED, the rest GENERATIVE.
"""

from __future__ import annotations

from .base import (NOVEL_VIEW_COMPLETION, SCENE_STATIC, SINGLE_VIEW, Availability, Backend, BackendContext,
                   BackendError, BackendResult, Estimate, JobInput, run_worker)
from .multiview import MultiViewBackend

TRAJECTORIES = {
    "arc": "around the subject: 45° to each side, from above and from below (best for people)",
    "orbit": "wide orbit around the main subject (±100°, above and below with Wan 2.2; 360° with Stable "
             "Virtual Camera)",
    "explore": "look around from where the photo was taken, with a little head movement (best for VR180)",
    "spiral": "small forward-facing spiral (most faithful, least new content)",
}
FRAMES = {"fast": 48, "auto": 80, "quality": 110}          # Stable Virtual Camera views
# (frames per shot (4k+1), sampling steps). Sharp fusion only uses a few key views per shot, so its
# shots can be shorter; a trained splat wants many frames.
WAN_SETTINGS = {"fusion": {"fast": (25, 25), "auto": (33, 30), "quality": (49, 40)},
                "train": {"fast": (49, 30), "auto": (81, 50), "quality": (81, 50)}}
FUSION_SEVA_VIEWS = 16
SEVA_MODELS = ("seva-1.1", "sd21-vae", "clip-vit-h-14")
WAN_MODELS = ("wan2.2-fun-5b-camera",)


def generation_engine(ctx, preferred: str | None = None) -> str | None:
    """'wan' or 'seva' (whichever is installed; Wan first), or None."""
    def ok(ids):
        return all(m in ctx.models.entries and ctx.models.is_installed(m) for m in ids)

    order = ["wan", "seva"] if preferred != "seva" else ["seva", "wan"]
    for e in order:
        if ok(WAN_MODELS if e == "wan" else SEVA_MODELS):
            return e
    return None


class GenerativeSceneBackend(Backend):
    id = "generative_scene"
    display_name = "Generative 3D from one photo (Wan 2.2 / Stable Virtual Camera + multi-view splats)"
    inputs = ("photo", "video:static_scene")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC, NOVEL_VIEW_COMPLETION})
    runtime_id = "gen-cu128"
    maturity = "experimental"
    upstream = ["videox-fun", "stable-virtual-camera", "vggt", "moge", "gsplat"]
    commercial_use = False  # VGGT-1B (reconstruction) and Stable Virtual Camera weights are non-commercial
    worker_script = "seva_worker.py"
    wan_script = "wan_worker.py"
    description = ("Generates the unseen sides of a photo with a camera-controlled diffusion model and trains a "
                   "real 3D Gaussian splat from the generated views. Invented content is labelled GENERATIVE.")

    def __init__(self):
        self.mv = MultiViewBackend()

    def model_ids(self, options: dict) -> list[str]:
        engine = options.get("engine") or getattr(self, "_engine", None) or "wan"
        return list(WAN_MODELS if engine == "wan" else SEVA_MODELS) + self.mv.model_ids(options)

    def prepare(self, ctx: BackendContext, options: dict) -> None:
        self._engine = generation_engine(ctx, options.get("engine"))
        super().prepare(ctx, {**options, "engine": self._engine or "wan"})

    def min_vram_gb(self, options: dict) -> float | None:
        return 12

    def availability(self, ctx: BackendContext, hw, options: dict | None = None) -> Availability:
        options = dict(options or {})
        options["engine"] = generation_engine(ctx, options.get("engine")) or "wan"
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
        engine = getattr(self, "_engine", None) or generation_engine(ctx, options.get("engine")) or "wan"
        mode = options.get("mode", "auto")
        assembly = options.get("assembly") or "fusion"
        if assembly not in WAN_SETTINGS:
            assembly = "fusion"
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        log = options.get("log") or (lambda m: None)
        out_dir = str(inp.work_dir / "generated_views")
        if engine == "wan":
            name, models = "Wan 2.2 Fun 5B Control-Camera", WAN_MODELS
            log(f"generating camera shots with {name} along a '{traj}' path: {TRAJECTORIES[traj]}")
            frames, steps = WAN_SETTINGS[assembly].get(mode, WAN_SETTINGS[assembly]["auto"])
            req = {"image": str(ref), "output_dir": out_dir, "trajectory": traj, "frames": frames,
                   "steps": steps, "short_side": 704, "hfov_deg": options.get("hfov_deg"),
                   "seed": int(options.get("seed", 42)), "model_dir": str(ctx.models.model_dir(WAN_MODELS[0])),
                   "every": 2}
            ctx.models.paths(WAN_MODELS[0])
            script = self.wan_script
        else:
            name, models = "Stable Virtual Camera", SEVA_MODELS
            n = FRAMES.get(mode, 80)
            log(f"generating {n} views with {name} along a '{traj}' path: {TRAJECTORIES[traj]}")
            req = {"image": str(ref), "output_dir": out_dir, "trajectory": traj, "num_frames": n, "steps": 50,
                   "short_side": 576, "hfov_deg": options.get("hfov_deg"), "seed": int(options.get("seed", 23)),
                   "seva_dir": str(ctx.models.model_dir("seva-1.1")), "vae_dir": str(ctx.models.model_dir("sd21-vae")),
                   "clip_path": str(ctx.models.paths("clip-vit-h-14")["open_clip_model.safetensors"])}
            for m in ("seva-1.1", "sd21-vae"):
                ctx.models.paths(m)  # clear error when missing
            script = self.worker_script
        # Wan: each memory setting runs in a fresh process (only an exiting process reliably frees VRAM
        # and RAM); the worker reports 'oom_retry' while a lighter setting is left, and reuses the shots
        # it already finished.
        for attempt in range(8):
            req["attempt"] = attempt
            try:
                out = run_worker(ctx.runtimes.python(self.runtime_id), script, req, inp.work_dir / "worker",
                                 progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id),
                                 log=options.get("log"), progress_range=(0.0, 0.5), timeout_s=4 * 3600)
                break
            except BackendError as e:
                if e.code != "oom_retry":
                    raise
                log(str(e))
        views = out["result"]["views"]
        if assembly == "fusion":
            views = self.fusion_views(views)
            log(f"sharp fusion of the photo and {len(views) - 1} key views")
        # The real photo is sampled more often than any single generated view during training.
        w_in = max(4.0, len(views) / 10)
        mv_views = [{"path": v["path"], "generated": bool(v["generated"]), "weight": 1.0 if v["generated"] else w_in}
                    for v in views]
        gen_models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                      for m in models]
        res = self.mv.reconstruct(
            mv_views, inp, ctx, {**options, "max_side": 1024, "assembly": assembly}, progress, cancel,
            progress_range=(0.5, 0.97), extra_models=gen_models,
            extra_warnings=[f"Generative completion: {len(views) - 1} views were invented by {name} "
                            f"('{traj}' path). Splats taken from those views are labelled GENERATIVE; they are "
                            "plausible, not measured."])
        res.backend = self.id
        res.extra.update({"trajectory": traj, "engine": engine, "assembly": assembly,
                          "generated_views": len(views) - 1,
                          "generation_vram_peak_mib": out["result"].get("vram_peak_mib")})
        res.worker_env = {"generation": out["env"], "reconstruction": res.worker_env}
        return res

    @staticmethod
    def fusion_views(views: list[dict]) -> list[dict]:
        """The photo plus the key views of every shot (Wan marks them; for engines that do not, an even
        selection of the generated views)."""
        photo = [v for v in views if not v.get("generated")][:1]
        gen = [v for v in views if v.get("generated")]
        keys = [v for v in gen if v.get("key")]
        if not keys and gen:
            n = min(FUSION_SEVA_VIEWS, len(gen))
            keys = [gen[round(i * (len(gen) - 1) / max(n - 1, 1))] for i in range(n)]
        return photo + keys
