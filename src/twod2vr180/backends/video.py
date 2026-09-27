"""Multi-view video backend (recon3d: VGGT + MoGe-2 metric + gsplat) and the
registry of evaluated-but-not-shippable research backends."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..scene import OBSERVED, read_gaussian_ply, write_gaussian_ply, write_splat
from .base import (MULTI_VIEW, NOVEL_VIEW_COMPLETION, SCENE_STATIC, SINGLE_VIEW, Backend, BackendContext,
                   BackendResult, Estimate, JobInput, run_worker)


class Recon3DBackend(Backend):
    id = "recon3d_video"
    display_name = "recon3d (VGGT + gsplat)"
    inputs = ("video:moving_camera",)
    capabilities = frozenset({MULTI_VIEW, SCENE_STATIC})
    runtime_id = "recon3d-cu124"
    maturity = "experimental"
    upstream = ["recon3d", "vggt", "moge", "gsplat"]
    commercial_use = False  # recon3d loads the non-commercial facebook/VGGT-1B checkpoint
    worker_script = "recon3d_worker.py"
    description = ("Moving-camera video → VGGT poses/depth (chunked, factor-graph refined) → MoGe-2 metric "
                   "scale → gsplat 3DGS training → .ply/.splat and TSDF .obj. Geometry is multi-view "
                   "OBSERVED for regions seen from several frames.")

    def model_ids(self, options: dict) -> list[str]:
        return ["vggt-1b", "moge-2-vitl"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 12

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        steps = 3000 if options.get("mode") == "fast" else 7000
        n = len(inp.frames)
        return Estimate(seconds=60 + n * 2 + steps * 0.05, vram_gb=12, disk_gb=1.0,
                        notes=["recon3d README: ~5 min for 15 images on an NVIDIA L4; not yet benchmarked on RTX 4080"])

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        img_dir = inp.work_dir / "tmp" / "recon3d_frames"
        img_dir.mkdir(parents=True, exist_ok=True)
        for p in inp.frames:
            shutil.copy2(p, img_dir / Path(p).name)
        out_dir = inp.work_dir / "recon3d_out"
        fast = options.get("mode") == "fast"
        req = {"image_dir": str(img_dir), "output_dir": str(out_dir), "max_frames": len(inp.frames),
               "steps": 3000 if fast else 7000, "resize": 720 if fast else 960, "mesh": True, "metric": True}
        out = run_worker(ctx.runtimes.python(self.runtime_id), self.worker_script, req,
                         inp.work_dir / "worker", progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id),
                         log=options.get("log"), progress_range=(0.0, 0.9), timeout_s=3 * 3600)
        res = out["result"]["outputs"]
        scene = read_gaussian_ply(Path(res["ply"]))
        # gsplat-optimised splats are fitted to multiple observations. Regions
        # seen in only one frame are not separable here; report as observed and
        # say so in the provenance note.
        scene.provenance[:] = OBSERVED
        cams = json.loads(Path(res["cameras"]).read_text()) if Path(res["cameras"]).exists() else []
        from ..scene import Camera
        scene.cameras = [Camera.from_dict(c) for c in cams]
        scene.metric_scale = bool(out["result"].get("metric"))
        exp = inp.work_dir / "export"
        ply = write_gaussian_ply(scene, exp / "scene.ply")
        splat = write_splat(scene, exp / "scene.splat")
        obj = None
        if res.get("obj"):
            obj = exp / "scene.obj"
            shutil.copy2(res["obj"], obj)
            mtl = Path(res["obj"]).with_suffix(".mtl")
            if mtl.exists():
                shutil.copy2(mtl, exp / mtl.name)
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=obj, cameras=cams,
            metric_scale=scene.metric_scale,
            provenance_note=("Splats optimised against multiple frames (OBSERVED). Surfaces seen from a single "
                             "frame are included without separate labelling; unseen regions are empty."),
            vram_peak_mib=out["result"].get("vram_peak_mib"), worker_env=out["env"],
            warnings=([] if out["result"].get("factor_graph", True) else
                      ["GTSAM is not available on Windows: long videos are aligned by chunk stitching only "
                       "(no factor-graph/loop-closure refinement); expect more drift on long camera paths."]),
            models_used=[{"id": "vggt-1b", "revision": "fetched by upstream", "license": "non-commercial"},
                         {"id": "moge-2-vitl", "revision": "fetched by upstream", "license": "UNVERIFIED"}],
            extra={"gaussians": len(scene)})


class ResearchOnlyBackend(Backend):
    """Evaluated upstream projects that cannot run in this build. Registered so
    the UI and reports can say *why* instead of silently omitting them."""

    maturity = "unsupported"

    def __init__(self, id: str, name: str, upstream: str, caps: frozenset[str], inputs: tuple[str, ...],
                 reason: str):
        self.id = id
        self.display_name = name
        self.upstream = [upstream]
        self.capabilities = caps
        self.inputs = inputs
        self.description = reason

    def run(self, *a, **k):  # pragma: no cover - availability() prevents this
        raise NotImplementedError(self.description)


RESEARCH_BACKENDS = [
    ResearchOnlyBackend(
        "one2scene", "One2Scene", "one2scene", frozenset({SINGLE_VIEW, NOVEL_VIEW_COMPLETION}), ("photo",),
        "Unsupported on RTX 4080 16 GB for now: needs a ~19 GB denoise checkpoint plus SDXL-VAE, ZIM and "
        "GFPGAN stacks, repository has no LICENSE file (all rights reserved by default), Linux/CUDA only."),
    ResearchOnlyBackend(
        "longsplat", "LongSplat", "longsplat", frozenset({MULTI_VIEW}), ("video:moving_camera",),
        "Unsupported: NVIDIA Source Code License (non-commercial) + Inria 3DGS licence; requires compiling "
        "simple-knn / diff-gaussian-rasterization / fused-ssim CUDA extensions (MSVC + CUDA toolkit on Windows)."),
    ResearchOnlyBackend(
        "depthsplat", "DepthSplat", "depthsplat", frozenset({MULTI_VIEW}), ("video:moving_camera",),
        "Not integrated yet: MIT code (cvg/depthsplat), but requires posed input views and a dataset-style "
        "loader; evaluate after the recon3d path is GPU-validated."),
    ResearchOnlyBackend(
        "gsfixer", "GSFixer", "gsfixer", frozenset({NOVEL_VIEW_COMPLETION}), (),
        "Unsupported: repository has no LICENSE file, tested only on H20/H100 (CUDA 12.1), requires compiled "
        "rasteriser extensions and a video-diffusion model."),
    ResearchOnlyBackend(
        "gsfix3d", "GSFix3D", "gsfix3d", frozenset({NOVEL_VIEW_COMPLETION}), (),
        "Research-only: Gaussian-Splatting licence makes the project non-commercial; models under OpenRAIL++-M; "
        "requires per-scene fine-tuning of a diffusion model and compiled rasteriser extensions."),
]
