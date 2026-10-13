"""Real 3D v3 (rc56): the photo locked.

Research (docs/research/single-image-splat.md, "Real 3D v3") found the common thread of every Real 3D
failure so far: AI views are drawn independently and are not 3D-consistent, and every view got its own
monocular depth. Nothing reconstructs one rigid 3D from that. The practice that works (3D Photo Inpainting,
WonderWorld, GaussVideoDreamer, GEN3C/Lyra) is the opposite: the photo's geometry rules everything it shows,
and new content only fills holes, with its depth tied to the known one.

This backend builds that foundation:
  1. the photo's lens (EXIF, else MoGe's field of view) and metric geometry (MoGe-3, else MoGe-2);
  2. a real-world scale check from the eyes of the people in the photo (YuNet; adults' eyes are 56-70 mm
     apart): the depth is rescaled only when it is clearly outside that range;
  3. Apple SHARP *with that depth injected* (workers/sharp_depth_worker.py): SHARP's sharp splats and its
     second layer behind edges, with MoGe's geometry and scale instead of SHARP's monocular guess;
  4. diagnostics: SHARP alone is exported next to it, and a sheet of views (head 15 cm to each side and up,
     15° around the subject) compares the two.
The progressive hole filling (rc57) builds on this splat.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..scene import INFERRED, Camera, GaussianScene, read_gaussian_ply, write_gaussian_ply, write_splat
from .base import (SCENE_STATIC, SINGLE_VIEW, Availability, Backend, BackendContext, BackendError, BackendResult,
                   Estimate, JobInput, run_worker)

MOGE_MODELS = ("moge-3-vitl", "moge-2-vitl-normal")     # first installed wins
FACE_MODEL = "yunet-face"
SHEET_VIEWS = [("head 15 cm left", "move", (-0.15, 0, 0)), ("head 15 cm right", "move", (0.15, 0, 0)),
               ("head 10 cm up", "move", (0, -0.10, 0)), ("15° left", "orbit", -15), ("15° right", "orbit", 15)]


def moge_model(ctx) -> str | None:
    for m in MOGE_MODELS:
        if m in ctx.models.entries and ctx.models.is_installed(m):
            return m
    return None


def sheet_cameras(cam: Camera, pivot: float, side: int = 480) -> list[tuple[str, dict]]:
    """The photo's camera, then the diagnostic views (photo frame, OpenCV)."""
    from ..fusion import orbit_pose

    s = side / max(cam.width, cam.height)
    base = {"width": max(8, round(cam.width * s)), "height": max(8, round(cam.height * s)), "fx": cam.fx * s,
            "fy": cam.fy * s, "cx": cam.cx * s, "cy": cam.cy * s}
    out = [("photo", {**base, "c2w": np.eye(4).tolist()})]
    for label, kind, v in SHEET_VIEWS:
        c2w = np.eye(4)
        if kind == "move":
            c2w[:3, 3] = v
        else:
            _, R, t = orbit_pose(v, 0, pivot)
            c2w[:3, :3], c2w[:3, 3] = R, t
        out.append((label, {**base, "c2w": c2w.tolist()}))
    return out


def pivot_depth(scene: GaussianScene, cam: Camera) -> float:
    from ..fusion import DepthBuffer

    buf = DepthBuffer(cam, 256).fill(scene.means, scene.opacity > 0.1)
    c = buf.depth[buf.h // 3: 2 * buf.h // 3, buf.w // 3: 2 * buf.w // 3]
    return float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else 2.0


def render_sheet(ctx, rows: list[tuple[str, Path, GaussianScene]], cams: list[tuple[str, dict]], work: Path,
                 out_png: Path, log=None) -> Path:
    """Grid of renders: one row per splat, one column per camera. GPU (gsplat) when it can run, else the
    CPU splat renderer."""
    from PIL import Image, ImageDraw

    from ..render import render_pinhole

    images = []
    for name, ply, scene in rows:
        row = None
        try:
            d = work / f"sheet_{name}"
            run_worker(ctx.runtimes.python("photo-cu128"), "render_worker.py",
                       {"ply": str(ply), "views": [c for _, c in cams], "out_dir": str(d), "batch": 6},
                       work / "worker", lambda v, m: None, lambda: False,
                       env=ctx.runtimes.worker_env("photo-cu128"), timeout_s=600)
            row = []
            for i in range(len(cams)):
                with np.load(d / f"view_{i:06d}.npz") as z:
                    row.append(z["rgb"])
        except (BackendError, OSError, KeyError) as e:
            if log:
                log(f"views sheet: GPU renderer unavailable ({str(e).splitlines()[0][:120]}); CPU renderer used")
            row = [render_pinhole(scene, np.array(c["c2w"]), c["width"], c["height"], c["fx"], c["fy"], c["cx"],
                                  c["cy"]).rgb for _, c in cams]
        images.append((name, row))
    w, h = cams[0][1]["width"], cams[0][1]["height"]
    pad = 22
    sheet = Image.new("RGB", (w * len(cams), (h + pad) * len(images) + pad), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    for j, (label, _) in enumerate(cams):
        dr.text((j * w + 6, 4), label, fill=(230, 230, 230))
    for i, (name, row) in enumerate(images):
        y = pad + i * (h + pad)
        dr.text((6, y + 2), name, fill=(255, 210, 90))
        for j, im in enumerate(row):
            sheet.paste(Image.fromarray(np.asarray(im, np.uint8)), (j * w, y + 16))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


class LockedPhotoBackend(Backend):
    id = "photo_locked"
    display_name = "Real 3D — the photo locked (Apple SHARP + measured depth)"
    inputs = ("photo", "video:frame", "video:static_scene")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC})
    runtime_id = "photo-cu128"
    maturity = "experimental"
    upstream = ["sharp", "moge"]
    commercial_use = False          # SHARP weights: research only
    geometry_script = "moge_worker.py"
    worker_script = "sharp_depth_worker.py"
    description = ("One photo → Apple SHARP's Gaussians aligned to the photo's measured metric depth (MoGe-3) at "
                   "its real lens, with a scale check from the people's eyes.")

    def model_ids(self, options: dict) -> list[str]:
        return ["sharp"]

    def min_vram_gb(self, options: dict) -> float | None:
        return 8

    def availability(self, ctx: BackendContext, hw, options: dict | None = None) -> Availability:
        av = super().availability(ctx, hw, options)
        if moge_model(ctx) is None:
            av.ok = False
            av.reasons.append("models not installed: MoGe-3 or MoGe-2 Large (the photo's measured depth)")
            av.missing_models.append(MOGE_MODELS[0])
        return av

    def estimate(self, inp: JobInput, hw, options: dict) -> Estimate:
        return Estimate(seconds=60, vram_gb=8, disk_gb=0.3, notes=["estimate; not yet benchmarked on RTX 4080"])

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        log = options.get("log") or (lambda m: None)
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        work = inp.work_dir / "locked"
        work.mkdir(parents=True, exist_ok=True)
        py, env = ctx.runtimes.python(self.runtime_id), ctx.runtimes.worker_env(self.runtime_id)
        models_used = []

        def used(mid):
            st = ctx.models.status(mid)
            models_used.append({"id": mid, **{k: st[k] for k in ("revision", "license", "hash_status")}})

        # 1. the photo's lens and metric geometry
        mg = moge_model(ctx)
        item = {"path": str(ref), "out": str(work / "photo.npz")}
        if options.get("hfov_deg"):
            item["fov_x"] = float(options["hfov_deg"])
        log(f"the photo's geometry: {'MoGe-3' if mg == MOGE_MODELS[0] else 'MoGe-2'}"
            + (f" at the EXIF lens ({item['fov_x']:.1f}°)" if "fov_x" in item else " (lens estimated from the image)"))
        run_worker(py, self.geometry_script,
                   {"model_path": str(ctx.models.paths(mg)["model.pt"]), "version": "v3" if mg == MOGE_MODELS[0] else "v2",
                    "refine_steps": 3, "max_side": 1536, "fp16": True, "images": [item],
                    "allow_cpu": bool(getattr(ctx, "allow_cpu", False))},
                   inp.work_dir / "worker", progress, cancel, env=env, log=log, progress_range=(0.0, 0.35),
                   timeout_s=1800)
        used(mg)

        # 2-3. scale check and SHARP with the measured depth
        face = None
        if FACE_MODEL in ctx.models.entries and ctx.models.is_installed(FACE_MODEL):
            face = str(next(iter(ctx.models.paths(FACE_MODEL).values())))
            used(FACE_MODEL)
        out = run_worker(py, self.worker_script,
                         {"mode": "photo", "checkpoint": str(ctx.models.paths("sharp")["sharp_2572gikvuh.pt"]),
                          "geometry": item["out"], "output_dir": str(work), "face_model": face, "plain": True,
                          "allow_cpu": bool(getattr(ctx, "allow_cpu", False))},
                         inp.work_dir / "worker", progress, cancel, env=env, log=log, progress_range=(0.35, 0.8),
                         timeout_s=1800)
        used("sharp")
        res = out["result"]
        scene = read_gaussian_ply(Path(res["outputs"]["locked"]))
        scene.provenance[:] = INFERRED
        scene.metric_scale = True
        if not scene.cameras:
            w, h = res["size"]
            scene.cameras = [Camera(w, h, res["f_px"], res["f_px"], w / 2, h / 2)]
        exp = inp.work_dir / "export"
        ply = write_gaussian_ply(scene, exp / "scene.ply", include_provenance=True)
        splat = write_splat(scene, exp / "scene.splat")
        extra_files = {}
        rows = [("photo locked (SHARP + measured depth)", ply, scene)]
        if res["outputs"].get("plain"):
            plain = read_gaussian_ply(Path(res["outputs"]["plain"]))
            plain.provenance[:] = INFERRED
            plain.cameras = scene.cameras[:1]
            p2 = write_gaussian_ply(plain, exp / "compare_sharp_alone.ply")
            extra_files["compare_sharp_alone"] = str(p2)
            rows.append(("SHARP alone (for comparison)", p2, plain))

        # 4. diagnostics sheet
        progress(0.85, "views sheet")
        try:
            cams = sheet_cameras(scene.cameras[0], pivot_depth(scene, scene.cameras[0]))
            sheet = render_sheet(ctx, rows, cams, work, exp / "views_sheet.png", log)
            extra_files["views_sheet"] = str(sheet)
        except Exception as e:  # noqa: BLE001 - a diagnostic
            log(f"views sheet skipped: {e}")
        progress(1.0, "done")
        (work / "summary.json").write_text(json.dumps({k: v for k, v in res.items() if k != "outputs"}, indent=2))
        notes = []
        if res.get("faces"):
            notes.append(f"scale checked on {len(res['faces'])} face(s)"
                         + (f", corrected ×{res['face_scale']:.3f}" if res.get("face_scale", 1.0) != 1.0 else ""))
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=None, cameras=[scene.cameras[0].to_dict()],
            metric_scale=True,
            provenance_note=("Splats INFERRED from the photo: Apple SHARP aligned to the measured depth "
                             f"({'MoGe-3' if mg == MOGE_MODELS[0] else 'MoGe-2'}). Nothing behind the subject is "
                             "invented yet."),
            vram_peak_mib=res.get("vram_peak_mib"), worker_env=out["env"], models_used=models_used,
            warnings=["OBJ mesh is not available for SHARP output."] + notes,
            extra={"gaussians": len(scene), "assembly": "locked", "trajectory": "tri",
                   "sharp_own_scale": res.get("sharp_own_scale"), "depth_coverage": res.get("depth_coverage"),
                   "faces": res.get("faces") or [], "face_scale": res.get("face_scale", 1.0),
                   "hfov_used_deg": float(np.degrees(2 * np.arctan(res["size"][0] / 2 / res["f_px"]))),
                   "files": extra_files})

    def export(self, result: BackendResult, destination: Path) -> list[Path]:
        out = super().export(result, destination)
        for p in (result.extra.get("files") or {}).values():
            if p and Path(p).exists():
                tgt = destination / Path(p).name
                if Path(p).resolve() != tgt.resolve():
                    shutil.copy2(p, tgt)
                out.append(tgt)
        return out
