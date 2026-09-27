"""Single-image backends: MoGe-2 and Depth-Anything-V2 RGB-D splats, Apple SHARP."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..rgbd import disparity_to_points, pointmap_to_gaussians, pointmap_to_obj
from ..scene import INFERRED, Camera, read_gaussian_ply, write_gaussian_ply, write_splat
from .base import (DYNAMIC, SCENE_STATIC, SINGLE_VIEW, Backend, BackendContext, BackendResult, Estimate,
                   JobInput, run_worker)


def load_pointmap(npz_path: Path, hfov_deg: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, Camera]:
    """Load a worker geometry file. Metric point maps (MoGe) are used as-is;
    relative disparity maps (Depth-Anything) are lifted with an assumed FOV."""
    d = np.load(npz_path)
    img = d["image"]
    h, w = img.shape[:2]
    if "points" in d.files:
        K = d["intrinsics"]
        cam = Camera(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        return d["points"], img, d["mask"], cam
    disp = d["disparity"]
    if disp.shape != (h, w):
        from PIL import Image

        disp = np.asarray(Image.fromarray(disp.astype(np.float32)).resize((w, h), Image.BILINEAR))
    pts, cam = disparity_to_points(disp, hfov_deg or 65.0)
    return pts, img, np.ones((h, w), bool), cam


class _RGBDBackend(Backend):
    """Shared path: worker writes per-frame geometry → Gaussians + OBJ."""

    inputs = ("photo", "video:static_scene", "video:static_camera_dynamic")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC, DYNAMIC})
    runtime_id = "photo-cu128"
    cpu_capable = True
    worker_script = ""
    metric = True

    def _request(self, ctx: BackendContext, options: dict, images: list[dict]) -> dict:
        raise NotImplementedError

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        model_id = self.model_ids(options)[0]
        geo = inp.work_dir / "geometry"
        geo.mkdir(parents=True, exist_ok=True)
        images = [{"path": str(p), "out": str(geo / f"{Path(p).stem}.npz")} for p in inp.frames]
        req = self._request(ctx, options, images)
        req["allow_cpu"] = bool(getattr(ctx, "allow_cpu", False))
        out = run_worker(ctx.runtimes.python(self.runtime_id), self.worker_script, req, inp.work_dir / "worker",
                         progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id),
                         log=options.get("log"), progress_range=(0.0, 0.8))
        npzs = [Path(p) for p in out["result"]["outputs"]]
        ref = npzs[min(int(options.get("reference_frame_index", 0)), len(npzs) - 1)]
        progress(0.85, "building Gaussians and mesh")
        hfov = options.get("hfov_deg")
        points, img, mask, cam = load_pointmap(ref, hfov)
        scene = pointmap_to_gaussians(points, img, mask, cam, metric=self.metric)
        exp = inp.work_dir / "export"
        ply = write_gaussian_ply(scene, exp / "scene.ply")
        splat = write_splat(scene, exp / "scene.splat")
        mesh_info = pointmap_to_obj(points, img, mask, exp / "scene.obj")
        progress(1.0, "done")
        st = ctx.models.status(model_id)
        warnings = []
        if not self.metric:
            warnings.append(f"Depth is relative (scale guessed from an assumed {hfov or 65.0:.0f}° field of view "
                            "and a 1–12 m depth range); stereo strength may need the eye-separation setting.")
        if out["env"].get("cuda_available") is False:
            warnings.append("Inference ran on the CPU (no CUDA device available to the runtime).")
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=exp / "scene.obj",
            frame_scenes=npzs if len(npzs) > 1 else [], cameras=[cam.to_dict()], metric_scale=self.metric,
            provenance_note="All splats INFERRED: observed colours, network-predicted depth; no hidden geometry.",
            vram_peak_mib=out["result"].get("vram_peak_mib"), worker_env=out["env"],
            models_used=[{"id": model_id, "revision": st["revision"], "license": st["license"],
                          "hash_status": st["hash_status"]}],
            warnings=warnings, extra={"mesh": mesh_info, "gaussians": len(scene)})


class MoGeRGBDBackend(_RGBDBackend):
    id = "moge_rgbd"
    display_name = "MoGe-2 (metric depth → splats + mesh)"
    maturity = "experimental"
    upstream = ["moge"]
    commercial_use = False  # until the MoGe-2 weight license is verified
    worker_script = "moge_worker.py"
    description = ("Monocular metric point map (MoGe-2) turned into surface-aligned Gaussians and a "
                   "textured mesh. Visible surfaces only: depth is INFERRED, nothing behind occluders "
                   "is invented, so large camera moves reveal holes.")

    def model_ids(self, options: dict) -> list[str]:
        return ["moge-2-vits-normal" if options.get("mode") == "fast" else "moge-2-vitl-normal"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 4 if options.get("mode") == "fast" else 6

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        n = max(1, len(inp.frames))
        per = 0.4 if options.get("mode") == "fast" else 1.0
        return Estimate(seconds=20 + n * per, vram_gb=self.min_vram_gb(options), disk_gb=0.05 * n,
                        notes=["estimate from upstream figures; not yet benchmarked on RTX 4080"])

    def _request(self, ctx, options, images):
        model_id = self.model_ids(options)[0]
        return {"model_path": str(ctx.models.paths(model_id)["model.pt"]),
                "max_side": 1024 if options.get("mode") == "fast" else 1536, "fp16": True, "images": images}


class DepthAnythingBackend(_RGBDBackend):
    id = "depth_anything_v2"
    display_name = "Depth-Anything-V2 Small (commercial-safe, relative depth)"
    maturity = "experimental"
    upstream = ["depth-anything-v2"]
    commercial_use = True  # Small checkpoint is Apache-2.0
    worker_script = "depth_anything_worker.py"
    metric = False
    description = ("Relative monocular depth (Depth-Anything-V2-Small, Apache-2.0) lifted to splats with an "
                   "assumed field of view (EXIF when available). Not metric: scale is a guess.")

    def model_ids(self, options: dict) -> list[str]:
        return ["depth-anything-v2-small"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 2

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        return Estimate(seconds=15 + 0.2 * max(1, len(inp.frames)), vram_gb=2, disk_gb=0.05 * len(inp.frames))

    def _request(self, ctx, options, images):
        model_dir = ctx.models.model_dir("depth-anything-v2-small")
        ctx.models.paths("depth-anything-v2-small")  # raises a clear error when missing
        return {"model_dir": str(model_dir), "max_side": 1024 if options.get("mode") == "fast" else 1536,
                "images": images}


class SharpBackend(Backend):
    id = "sharp"
    display_name = "Apple SHARP (research licence)"
    inputs = ("photo", "video:static_scene")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC})
    runtime_id = "photo-cu128"
    maturity = "research"
    upstream = ["sharp"]
    commercial_use = False
    cpu_capable = True
    description = ("Feed-forward single-image 3D Gaussians (Apple ml-sharp). Model weights are licensed "
                   "for non-commercial research only and are never bundled.")

    def model_ids(self, options: dict) -> list[str]:
        return ["sharp"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 8

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        return Estimate(seconds=30, vram_gb=8, disk_gb=0.2, notes=["not yet benchmarked on RTX 4080"])

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        ckpt = ctx.models.paths("sharp")["sharp_2572gikvuh.pt"]
        src = inp.work_dir / "tmp" / "sharp_in"
        dst = inp.work_dir / "sharp_out"
        src.mkdir(parents=True, exist_ok=True)
        dst.mkdir(parents=True, exist_ok=True)
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        shutil.copy2(ref, src / Path(ref).name)
        out = run_worker(ctx.runtimes.python(self.runtime_id), "sharp_worker.py",
                         {"checkpoint": str(ckpt), "input_dir": str(src), "output_dir": str(dst),
                          "allow_cpu": bool(getattr(ctx, "allow_cpu", False))},
                         inp.work_dir / "worker", progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id),
                         log=options.get("log"), progress_range=(0.0, 0.85))
        scene = read_gaussian_ply(Path(out["result"]["outputs"][0]))
        # SHARP predicts Gaussians only for what the image shows; depth is inferred.
        scene.provenance[:] = INFERRED
        scene.metric_scale = True
        exp = inp.work_dir / "export"
        ply = write_gaussian_ply(scene, exp / "scene.ply")
        splat = write_splat(scene, exp / "scene.splat")
        progress(1.0, "done")
        st = ctx.models.status("sharp")
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=None,
            cameras=[c.to_dict() for c in scene.cameras], metric_scale=True,
            provenance_note="Splats INFERRED from a single image (SHARP). No mesh export (not provided upstream).",
            worker_env=out["env"],
            models_used=[{"id": "sharp", "revision": st["revision"], "license": st["license"],
                          "hash_status": st["hash_status"]}],
            warnings=["OBJ mesh is not available for SHARP output."],
            extra={"gaussians": len(scene)})
