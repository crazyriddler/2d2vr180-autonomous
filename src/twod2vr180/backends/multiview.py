"""Multi-view Gaussian splatting: several photos of the same scene, the frames of
a moving-camera video, or views produced by a generative model.

VGGT (camera poses + depth, chunk-stitched) → MoGe-2 metric scale → 3DGS
optimisation with gsplat → per-splat provenance from real-view visibility.
Runs in the 'recon3d-cu124' runtime (torch 2.4 + prebuilt gsplat for Windows).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..scene import Camera, read_gaussian_ply, write_gaussian_ply, write_splat
from .base import (MULTI_VIEW, SCENE_STATIC, Backend, BackendContext, BackendResult, Estimate, JobInput,
                   run_worker)

QUALITY = {  # mode → (training steps, longest image side, splat cap)
    "fast": (5000, 720, 1_000_000),
    "auto": (9000, 960, 2_000_000),
    "quality": (15000, 1280, 3_000_000),
}
DA3_MODEL = "da3-nested-giant-large"   # preferred camera engine when installed (VGGT otherwise)
DA3_RES = {"fast": 392, "auto": 504, "quality": 616}   # processing size (multiples of 14)
FUSION = {  # sharp fusion: mode → (photo side, generated-view side, splat cap)
    "fast": (1024, 768, 2_000_000),
    "auto": (1536, 1024, 3_000_000),
    "quality": (2048, 1280, 4_000_000),
}


class MultiViewBackend(Backend):
    id = "multiview"
    display_name = "Multi-view 3D Gaussian Splatting (VGGT + gsplat)"
    inputs = ("images", "video:moving_camera")
    capabilities = frozenset({MULTI_VIEW, SCENE_STATIC})
    runtime_id = "recon3d-cu124"
    maturity = "experimental"
    upstream = ["vggt", "moge", "gsplat"]
    commercial_use = False  # facebook/VGGT-1B checkpoint is non-commercial
    worker_script = "multiview_worker.py"
    description = ("Several photos of one scene, or the frames of a video where the camera moves → VGGT camera "
                   "poses and depth → metric scale (MoGe-2) → trained 3D Gaussian splats and a TSDF mesh. "
                   "Surfaces seen from two or more photos are OBSERVED; the more angles, the more complete.")

    def model_ids(self, options: dict) -> list[str]:
        return ["vggt-1b", "moge-2-vitl-normal"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 10

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        steps, side, _ = QUALITY.get(options.get("mode", "auto"), QUALITY["auto"])
        n = max(len(inp.frames), 2)
        return Estimate(seconds=40 + n * 1.5 + steps * 0.03 * (side / 960) ** 2, vram_gb=10, disk_gb=0.5 + 0.01 * n,
                        notes=["estimate; not yet benchmarked on RTX 4080"])

    # ------------------------------------------------------------------ run
    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        frames = list(inp.frames)
        ref = min(int(options.get("reference_frame_index", 0)), len(frames) - 1)
        frames.insert(0, frames.pop(ref))  # the reference view defines the scene frame
        views = [{"path": str(p), "generated": False, "weight": 1.0} for p in frames]
        return self.reconstruct(views, inp, ctx, options, progress, cancel)

    def reconstruct(self, views: list[dict], inp: JobInput, ctx: BackendContext, options: dict, progress, cancel,
                    progress_range: tuple[float, float] = (0.0, 0.95), extra_models: list[dict] | None = None,
                    extra_warnings: list[str] | None = None) -> BackendResult:
        mode = options.get("mode", "auto")
        steps, side, cap = QUALITY.get(mode, QUALITY["auto"])
        assembly = options.get("assembly") if options.get("assembly") == "fusion" else "train"
        out_dir = inp.work_dir / "multiview_out"
        req = {"images": views, "output_dir": str(out_dir),
               "vggt_dir": str(ctx.models.model_dir("vggt-1b")),
               "moge_path": str(ctx.models.paths("moge-2-vitl-normal")["model.pt"]),
               "max_side": int(options.get("max_side", side)), "steps": int(options.get("train_steps", steps)),
               "max_gaussians": cap, "mesh": bool(options.get("mesh", True)), "assembly": assembly}
        da3 = DA3_MODEL in ctx.models.entries and ctx.models.is_installed(DA3_MODEL) and \
            options.get("camera_engine", "auto") != "vggt"
        if da3:
            req.update(pose_engine="da3", da3_dir=str(ctx.models.model_dir(DA3_MODEL)),
                       da3_res=DA3_RES.get(mode, 504))
        if assembly == "fusion":
            ref_side, gen_side, req["max_gaussians"] = FUSION.get(mode, FUSION["auto"])
            req.update(fuse_ref_side=ref_side, fuse_side=gen_side)
        ctx.models.paths("vggt-1b")  # clear error when missing
        out = run_worker(ctx.runtimes.python(self.runtime_id), self.worker_script, req, inp.work_dir / "worker",
                         progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id), log=options.get("log"),
                         progress_range=progress_range, timeout_s=4 * 3600)
        res = out["result"]
        files = res["outputs"]
        scene = read_gaussian_ply(Path(files["ply"]))
        cams = json.loads(Path(files["cameras"]).read_text()) if Path(files["cameras"]).exists() else []
        scene.cameras = [Camera.from_dict(c) for c in cams if not c.get("generated")] or \
            [Camera.from_dict(c) for c in cams[:1]]
        scene.metric_scale = bool(res.get("metric"))
        exp = inp.work_dir / "export"
        ply = write_gaussian_ply(scene, exp / "scene.ply")
        splat = write_splat(scene, exp / "scene.splat")
        obj = None
        if files.get("obj") and Path(files["obj"]).exists():
            obj = exp / "scene.obj"
            shutil.copy2(files["obj"], obj)
        p = res.get("provenance") or {}
        if res.get("assembly") == "fusion":
            note = (f"Sharp fusion of {res.get('views')} views: splats from the photo's own pixels are INFERRED "
                    f"(observed colours, network-predicted depth; {p.get('inferred', 0):,}); surfaces added from "
                    f"generated views are GENERATIVE ({p.get('generative', 0):,}).")
        else:
            note = (f"Trained on {res.get('views')} views ({res.get('real_views')} real). Splats seen by two or more "
                    f"real views are OBSERVED ({p.get('observed', 0):,}), by one real view INFERRED "
                    f"({p.get('inferred', 0):,})")
            note += (f", only by generated views GENERATIVE ({p.get('generative', 0):,})." if p.get("generative")
                     else ".")
        warnings = list(extra_warnings or [])
        if not res.get("metric"):
            warnings.append("Metric scale could not be estimated; stereo depth may need the eye-separation setting.")
        used = list(self.model_ids(options)) + ([DA3_MODEL] if res.get("camera_engine") == "da3" else [])
        models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                  for m in used]
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=obj,
            cameras=[c for c in cams if not c.get("generated")],
            metric_scale=scene.metric_scale, provenance_note=note, vram_peak_mib=res.get("vram_peak_mib"),
            worker_env=out["env"], models_used=models + list(extra_models or []), warnings=warnings,
            extra={"gaussians": len(scene), "views": res.get("views"), "real_views": res.get("real_views"),
                   "reference_psnr_db": res.get("reference_psnr_db"), "mesh": res.get("mesh"),
                   "assembly": res.get("assembly", "train"), "camera_engine": res.get("camera_engine", "vggt"),
                   "candidate_selection": res.get("candidate_selection") or [],
                   "metric_scale_factor": res.get("metric_scale_factor")})
