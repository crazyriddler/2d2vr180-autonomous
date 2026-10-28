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
2.0.0-rc7 "precision" (workers/precision.py, refine.py): the EXIF lens checked against MoGe-3's and Depth
Pro's own estimates; MoGe-3 several times (photo, mirror image, coarser level) joined to Apple Depth Pro; fine
relief from surface normals (MoGe-3's; Sapiens2's for people); depth edges sent to one side; every splat the
photo sees placed at that depth; the splat fitted to the photo from its own camera (colours, opacities,
sizes); relief.png shows each step. MoGe-3 Giant replaces the Large model for photos when installed.
Then (rc57) the holes are filled, consistent with the photo by construction:
  5. the locked splat is rendered from moved cameras (head 15 cm to each side, 15° around the subject; Auto and
     Quality go further). A hole is every pixel showing nothing the photo saw: empty, or SHARP's own guesses
     behind the photo's surfaces (fusion.unseen_by_photo);
  6. only the holes are painted (Qwen-Image-Edit with the holes blacked out, else LaMa); everything visible
     stays exactly the render;
  7. SHARP runs on each filled view with the rendered depth where the scene exists and MoGe's depth of the
     new pixels tied to the background around each hole (sharp_depth_worker "fill");
  8. only the splats in the holes are added (GENERATIVE), never on or in front of what the photo shows; the
     photo's own guesses where new content lands are replaced (fusion.add_fills).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..scene import GENERATIVE, INFERRED, Camera, GaussianScene, read_gaussian_ply, write_gaussian_ply, write_splat
from .base import (SCENE_STATIC, SINGLE_VIEW, Availability, Backend, BackendContext, BackendError, BackendResult,
                   Estimate, JobInput, run_worker)

MOGE_MODELS = ("moge-3-vitg", "moge-3-vitl")   # rc7: the Giant model for photos when installed
FACE_MODEL = "yunet-face"
HUMAN_SEG = "sapiens2-seg-0.4b"            # rc58: people's outline and 3D shape (Sapiens2)
HUMAN_POINTMAP = "sapiens2-pointmap-1b"
HUMAN_NORMAL = "sapiens2-normal-1b"        # rc7: people's fine relief
DEPTH_PRO = "depth-pro"                    # rc7: a second measured depth (sharp edges, lens check)
RENDER_RUNTIME = "render-cu124"            # rc8: gsplat with prebuilt CUDA kernels (the photo engine has none)
# "move": head translation (m); "orbit": (azimuth, elevation) degrees around the subject; "turn": (yaw, pitch)
# degrees looking around from the camera's place (positive: right, up) - what is outside the photo's frame.
_TURNS = [("look 40° left", "turn", (-40, 0)), ("look 40° right", "turn", (40, 0)),
          ("look 25° up", "turn", (0, 25)), ("look 25° down", "turn", (0, -25))]
SHEET_VIEWS = [("head 15 cm left", "move", (-0.15, 0, 0)), ("head 15 cm right", "move", (0.15, 0, 0)),
               ("head 10 cm up", "move", (0, -0.10, 0)), ("15° left", "orbit", (-15, 0)), ("15° right", "orbit", (15, 0)),
               _TURNS[0], _TURNS[1]]
# Moved cameras whose holes are filled, small moves first (an earlier view's content wins where they overlap);
# then (2.0.0-rc5) looking around from the camera's place, so in VR there is scene beyond the photo's frame.
_FAST = [("head 15 cm left", "move", (-0.15, 0, 0)), ("head 15 cm right", "move", (0.15, 0, 0)),
         ("15° left", "orbit", (-15, 0)), ("15° right", "orbit", (15, 0))]
_AUTO = _FAST + [("head 10 cm up", "move", (0, -0.10, 0)), ("30° left", "orbit", (-30, 0)), ("30° right", "orbit", (30, 0))]
FILL_VIEWS = {"fast": _FAST + _TURNS[:2], "auto": _AUTO + _TURNS,
              "quality": _AUTO + [("head 10 cm down", "move", (0, 0.10, 0)), ("15° from above", "orbit", (0, 15)),
                                  ("head 30 cm left", "move", (-0.30, 0, 0)), ("head 30 cm right", "move", (0.30, 0, 0))]
              + _TURNS}
FILL_PROMPT = ("The pure black areas are missing parts of this picture. Fill them with what is hidden behind, "
               "continuing the surrounding background, surfaces, texture and lighting seamlessly. Do not add new "
               "objects or people. Keep everything else exactly the same.")
MIN_HOLE = 0.002            # views with less of their picture in holes are skipped


def moge_model(ctx) -> str | None:
    for m in MOGE_MODELS:
        if m in ctx.models.entries and ctx.models.is_installed(m):
            return m
    return None


def moge_name(mg: str) -> str:
    return {"moge-3-vitg": "MoGe-3 Giant", "moge-3-vitl": "MoGe-3"}.get(mg, "MoGe-2")


def sheet_cameras(cam: Camera, pivot: float, side: int = 480, views=None, photo: bool = True) -> list[tuple[str, dict]]:
    """The photo's camera (optional), then the moved views (photo frame, OpenCV): "move" = head translation
    in metres, "orbit" = (azimuth, elevation) degrees around the subject at depth ``pivot``."""
    from ..fusion import orbit_pose

    s = min(side, max(cam.width, cam.height)) / max(cam.width, cam.height)
    base = {"width": max(8, round(cam.width * s)), "height": max(8, round(cam.height * s)), "fx": cam.fx * s,
            "fy": cam.fy * s, "cx": cam.cx * s, "cy": cam.cy * s}
    out = [("photo", {**base, "c2w": np.eye(4).tolist()})] if photo else []
    for label, kind, v in SHEET_VIEWS if views is None else views:
        c2w = np.eye(4)
        if kind == "move":
            c2w[:3, 3] = v
        elif kind == "turn":                 # OpenCV axes: yaw about y (right +), pitch about x (up +)
            yw, pt = np.radians(v[0]), np.radians(v[1])
            ry = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
            rx = np.array([[1, 0, 0], [0, np.cos(pt), -np.sin(pt)], [0, np.sin(pt), np.cos(pt)]])
            c2w[:3, :3] = ry @ rx
        else:
            _, R, t = orbit_pose(v[0], v[1], pivot)
            c2w[:3, :3], c2w[:3, 3] = R, t
        out.append((label, {**base, "c2w": c2w.tolist()}))
    return out


def view_camera(c: dict) -> Camera:
    return Camera(int(c["width"]), int(c["height"]), c["fx"], c["fy"], c["cx"], c["cy"], np.array(c["c2w"]))


def grid(rows: list[tuple[str, list]], col_labels: list[str], out_png: Path) -> Path:
    """Images in a labelled grid (rows of equal-size images)."""
    from PIL import Image, ImageDraw

    h, w = np.asarray(rows[0][1][0]).shape[:2]
    pad = 22
    sheet = Image.new("RGB", (w * len(col_labels), (h + pad) * len(rows) + pad), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    for j, label in enumerate(col_labels):
        dr.text((j * w + 6, 4), label, fill=(230, 230, 230))
    for i, (name, row) in enumerate(rows):
        y = pad + i * (h + pad)
        dr.text((6, y + 2), name, fill=(255, 210, 90))
        for j, im in enumerate(row):
            sheet.paste(Image.fromarray(np.asarray(im, np.uint8)).resize((w, h)), (j * w, y + 16))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def render_views(ctx, ply: Path, scene: GaussianScene, cams: list[dict], out_dir: Path, work: Path,
                 script: str = "render_worker.py", log=None, cpu_fallback: bool = True) -> list[dict] | None:
    """RGB, alpha and depth of a splat seen from cameras: gsplat in the renderer runtime (rc8: its prebuilt
    CUDA kernels; in the photo engine gsplat needs a CUDA toolkit to build them and never ran), else the CPU
    splat renderer. None when neither can run (``cpu_fallback`` False)."""
    from ..render import render_pinhole

    try:
        if not ctx.runtimes.is_installed(RENDER_RUNTIME):
            raise BackendError("the Real 3D renderer is not installed (Components → Install recommended)",
                               code="runtime_missing")
        run_worker(ctx.runtimes.python(RENDER_RUNTIME), script,
                   {"ply": str(ply), "views": cams, "out_dir": str(out_dir), "batch": 6}, work / "worker",
                   lambda v, m: None, lambda: False, env=ctx.runtimes.worker_env(RENDER_RUNTIME), timeout_s=900)
        out = []
        for i in range(len(cams)):
            with np.load(out_dir / f"view_{i:06d}.npz") as z:
                out.append({"rgb": z["rgb"], "alpha": z["alpha"].astype(np.float32),
                            "depth": z["depth"].astype(np.float32)})
        return out
    except (BackendError, OSError, KeyError) as e:
        if log:
            log(f"GPU renderer unavailable ({str(e).splitlines()[0][:160]})"
                + ("; CPU renderer used for the pictures" if cpu_fallback else ""))
        if not cpu_fallback:
            return None
        res = []
        for c in cams:
            r = render_pinhole(scene, np.array(c["c2w"]), c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"])
            res.append({"rgb": r.rgb, "alpha": r.covered.astype(np.float32), "depth": r.depth})
        return res


def pivot_depth(scene: GaussianScene, cam: Camera) -> float:
    from ..fusion import DepthBuffer

    buf = DepthBuffer(cam, 256).fill(scene.means, scene.opacity > 0.1)
    c = buf.depth[buf.h // 3: 2 * buf.h // 3, buf.w // 3: 2 * buf.w // 3]
    return float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else 2.0


def render_sheet(ctx, rows: list[tuple[str, Path, GaussianScene]], cams: list[tuple[str, dict]], work: Path,
                 out_png: Path, log=None, script: str = "render_worker.py") -> Path:
    """Grid of renders: one row per splat, one column per camera."""
    images = []
    for k, (name, ply, scene) in enumerate(rows):
        r = render_views(ctx, ply, scene, [c for _, c in cams], work / f"sheet_{k}", work, script, log)
        images.append((name, [v["rgb"] for v in r]))
    return grid(images, [label for label, _ in cams], out_png)


class LockedPhotoBackend(Backend):
    id = "photo_locked"
    display_name = "Real 3D — the photo locked (Apple SHARP + measured depth)"
    inputs = ("photo",)
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC})
    runtime_id = "photo-cu128"
    maturity = "experimental"
    upstream = ["sharp", "moge"]
    commercial_use = False          # SHARP weights: research only
    geometry_script = "moge_worker.py"
    worker_script = "sharp_depth_worker.py"
    render_script = "render_worker.py"
    refine_script = "refine_worker.py"
    painter_script = None             # default: backends/painter.SCRIPT (tests swap in a fake)
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
            av.reasons.append("models not installed: MoGe-3 Large (the photo's measured depth)")
            av.missing_models.append(MOGE_MODELS[-1])
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

        def installed_file(mid):
            if mid in ctx.models.entries and ctx.models.is_installed(mid):
                used(mid)
                return str(next(iter(ctx.models.paths(mid).values())))
            return None

        # 1. the photo's lens and metric geometry (rc7 "precision": lens check, several passes, Depth Pro)
        mg = moge_model(ctx)
        version = "v3" if mg.startswith("moge-3") else "v2"
        precise = bool(options.get("precision", True))
        item = {"path": str(ref), "out": str(work / "photo.npz")}
        if options.get("hfov_deg"):
            item["fov_x"] = float(options["hfov_deg"])
        log(f"the photo's geometry: {moge_name(mg)}"
            + (f" at the EXIF lens ({item['fov_x']:.1f}°)" if "fov_x" in item else " (lens estimated from the image)"))
        geo_req = {"model_path": str(ctx.models.paths(mg)["model.pt"]), "version": version,
                   "refine_steps": 3, "max_side": 1536, "fp16": True, "images": [item],
                   "allow_cpu": bool(getattr(ctx, "allow_cpu", False))}
        if precise:
            dp = installed_file(DEPTH_PRO)
            if not dp:
                log("a second measured depth needs Apple Depth Pro (Components); MoGe alone is used")
            geo_req["precision"] = {"passes": True, "depth_pro": dp, "lens_check": True}
        geo_out = run_worker(py, self.geometry_script, geo_req, inp.work_dir / "worker", progress, cancel, env=env,
                             log=log, progress_range=(0.0, 0.25), timeout_s=1800)
        used(mg)
        geo_prec = ((geo_out.get("result") or {}).get("precision") or [None])[0]

        # 2-3. scale check and SHARP with the measured depth

        face = installed_file(FACE_MODEL)
        human_seg, human_pm = installed_file(HUMAN_SEG), installed_file(HUMAN_POINTMAP)
        if not (human_seg and human_pm):
            human_seg = human_pm = None
            log("people's own 3D shape needs Sapiens2 (Components); the scene's depth is used for them")
        human_normal = installed_file(HUMAN_NORMAL) if precise and human_seg else None
        ipd = float(options.get("ipd_mm") or 63.0)
        detail = bool(options.get("detail"))
        if detail:
            log("maximum detail: the photo at up to 3072 px (4608 px from 24 MP), fine relief from MoGe-3 on parts "
                "of it, SHARP on parts")
        out = run_worker(py, self.worker_script,
                         {"mode": "photo", "checkpoint": str(ctx.models.paths("sharp")["sharp_2572gikvuh.pt"]),
                          "geometry": item["out"], "output_dir": str(work), "face_model": face, "plain": True,
                          "ipd_target_m": ipd / 1000.0, "human_seg": human_seg, "human_pointmap": human_pm,
                          "detail": detail, "photo": str(ref), "moge_path": str(ctx.models.paths(mg)["model.pt"]),
                          "moge_version": version, "precision": precise, "refine": False,
                          "human_normal": human_normal, "allow_cpu": bool(getattr(ctx, "allow_cpu", False))},
                         inp.work_dir / "worker", progress, cancel, env=env, log=log, progress_range=(0.25, 0.45),
                         timeout_s=3600)
        used("sharp")
        res = out["result"]
        if precise:                     # the splat fitted to the photo, in the renderer runtime (rc8)
            fit = self._fit(ctx, inp, ref, res, work, progress, cancel, log)
            if fit.get("used") and fit.get("ply"):
                res["outputs"]["locked"] = fit.pop("ply")
            if isinstance(res.get("precision"), dict):
                res["precision"]["fit"] = fit
        scene = read_gaussian_ply(Path(res["outputs"]["locked"]))
        scene.provenance[:] = INFERRED
        scene.metric_scale = True
        if not scene.cameras:
            w, h = res["size"]
            scene.cameras = [Camera(w, h, res["f_px"], res["f_px"], w / 2, h / 2)]
        exp = inp.work_dir / "export"
        cam = scene.cameras[0]
        extra_people = {}
        if (res.get("people") or {}).get("sheet") and Path(res["people"]["sheet"]).exists():
            exp.mkdir(parents=True, exist_ok=True)
            shutil.copy2(res["people"]["sheet"], exp / "people_shape.png")
            extra_people["people_shape"] = str(exp / "people_shape.png")
        if res["outputs"].get("relief") and Path(res["outputs"]["relief"]).exists():
            exp.mkdir(parents=True, exist_ok=True)
            shutil.copy2(res["outputs"]["relief"], exp / "relief.png")
            extra_people["relief"] = str(exp / "relief.png")
        locked_ply = write_gaussian_ply(scene, work / "locked_scene.ply", include_provenance=True)
        extra_files, fill_report, final = dict(extra_people), [], scene
        warnings = ["OBJ mesh is not available for SHARP output."]
        lens = (geo_prec or {}).get("lens") or {}
        if str(lens.get("from", "")).startswith("models (EXIF"):
            warnings.append("The photo's lens: " + lens.get("note", "the EXIF lens disagreed with the depth models"))

        # 5-8. fill the holes of moved views
        fill_status = "off (option turned off)"
        if options.get("real3d_fill", True):
            self.fill_note = None
            try:
                filled = self._fill(inp, ctx, options, progress, cancel, log, scene, locked_ply, cam, mg, used)
            except BackendError as e:
                if e.code == "cancelled":
                    raise
                filled = None
                self.fill_note = f"failed: {str(e).splitlines()[0][:200]}"
            if filled is None:
                fill_status = self.fill_note or "nothing to fill"
                warnings.append(f"AI fill of what the photo hides: {fill_status}. The photo's splat is used as it is.")
                log(warnings[-1])
            if filled is not None:
                final, added, fill_report, steps = filled
                extra_files["added_layer"] = str(write_gaussian_ply(added, exp / "added_layer.ply",
                                                                    include_provenance=True))
                if steps:
                    extra_files["fill_steps"] = str(steps)
        final.cameras = [cam]
        ply = write_gaussian_ply(final, exp / "scene.ply", include_provenance=True)
        splat = write_splat(final, exp / "scene.splat")
        rows = [("Real 3D" + (" (photo locked + filled holes)" if final is not scene else " (photo locked)"),
                 ply, final)]
        if final is not scene:
            rows.append(("photo locked, no fill", locked_ply, scene))
        if res["outputs"].get("standard"):          # maximum detail: the normal result next to it
            std = read_gaussian_ply(Path(res["outputs"]["standard"]))
            std.provenance[:] = INFERRED
            std.cameras = [cam]
            p3 = write_gaussian_ply(std, exp / "compare_normal_detail.ply")
            extra_files["compare_normal_detail"] = str(p3)
            rows.append(("normal detail (for comparison)", p3, std))
        if res["outputs"].get("plain"):
            plain = read_gaussian_ply(Path(res["outputs"]["plain"]))
            plain.provenance[:] = INFERRED
            plain.cameras = [cam]
            p2 = write_gaussian_ply(plain, exp / "compare_sharp_alone.ply")
            extra_files["compare_sharp_alone"] = str(p2)
            rows.append(("SHARP alone (for comparison)", p2, plain))

        # diagnostics sheet
        progress(0.93, "views sheet")
        try:
            cams = sheet_cameras(cam, pivot_depth(scene, cam))
            sheet = render_sheet(ctx, rows, cams, work, exp / "views_sheet.png", log, self.render_script)
            extra_files["views_sheet"] = str(sheet)
        except Exception as e:  # noqa: BLE001 - a diagnostic
            log(f"views sheet skipped: {e}")
        progress(1.0, "done")
        (work / "summary.json").write_text(json.dumps({k: v for k, v in res.items() if k != "outputs"}, indent=2))
        if res.get("scale_from") == "eye distance":
            n_eyes = sum(1 for f in res["faces"] if f.get("used", True))
            warnings.append(f"Real-world scale set from the eyes of {n_eyes} face(s) "
                            f"({res.get('ipd_target_m', 0.063) * 1000:.1f} mm apart): ×{res.get('scale', 1.0):.3f}.")
        elif res.get("scale_from") == "human model":
            warnings.append(f"Real-world scale set from the human model's distance of the people: "
                            f"×{res.get('scale', 1.0):.3f}.")
        added_n = int((final.provenance == GENERATIVE).sum())
        if final is not scene:
            views_added = [r for r in fill_report if r.get("view") != "photo" and r.get("added")]
            fill_status = (f"{added_n:,} AI-painted splats added from {len(views_added)} of "
                           f"{len([r for r in fill_report if r.get('view') != 'photo'])} views")
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=None, cameras=[cam.to_dict()], metric_scale=True,
            provenance_note=(f"{len(final) - added_n:,} splats INFERRED from the photo: Apple SHARP aligned to the "
                             f"measured depth ({moge_name(mg)}"
                             + (" + Depth Pro" if ((geo_prec or {}).get("depth_pro") or {}).get("used") else "")
                             + "). "
                             + (f"{added_n:,} GENERATIVE splats fill what the photo hides (AI-painted, their depth "
                                "tied to the photo's)." if added_n else "Nothing behind the subject is invented.")),
            vram_peak_mib=res.get("vram_peak_mib"), worker_env=out["env"], models_used=models_used,
            warnings=warnings,
            extra={"gaussians": len(final), "assembly": "locked", "trajectory": "tri", "added_splats": added_n,
                   "sharp_own_scale": res.get("sharp_own_scale"), "depth_coverage": res.get("depth_coverage"),
                   "faces": res.get("faces") or [], "face_scale": res.get("face_scale", 1.0),
                   "scale": res.get("scale", 1.0), "scale_from": res.get("scale_from"),
                   "ipd_target_mm": round(res.get("ipd_target_m", 0.063) * 1000, 1), "people": res.get("people"),
                   "hfov_used_deg": float(np.degrees(2 * np.arctan(res["size"][0] / 2 / res["f_px"]))),
                   "fill": fill_report, "fill_status": fill_status, "detail": res.get("detail"),
                   "precision": {"geometry": geo_prec, "photo": res.get("precision")} if precise else None,
                   "files": extra_files})

    def _fit(self, ctx, inp, ref, res, work, progress, cancel, log) -> dict:
        """refine_worker in the renderer runtime: the splat's colours, opacities and sizes fitted to the photo.
        Returns its stats (+ "ply" when kept); never fails the job."""
        if not ctx.runtimes.is_installed(RENDER_RUNTIME):
            log("fit to the photo skipped: it needs the Real 3D renderer (Components → Install recommended)")
            return {"used": False, "reason": "the Real 3D renderer is not installed"}
        src = Path(res["outputs"]["locked"])
        out = src.with_name(src.stem + "_fitted.ply")
        try:
            r = run_worker(ctx.runtimes.python(RENDER_RUNTIME), self.refine_script,
                           {"ply": str(src), "out": str(out), "photo": str(ref), "fx": float(res["f_px"]),
                            "size": list(res["size"]), "iters": 300},
                           inp.work_dir / "worker", progress, cancel, env=ctx.runtimes.worker_env(RENDER_RUNTIME),
                           log=log, progress_range=(0.45, 0.47), timeout_s=1800)
        except BackendError as e:
            if e.code == "cancelled":
                raise
            log(f"fit to the photo skipped ({str(e).splitlines()[0][:200]})")
            return {"used": False, "error": str(e).splitlines()[0][:300]}
        st = dict(r["result"].get("stats") or {})
        if st.get("used") and (r["result"].get("outputs") or {}).get("fitted"):
            st["ply"] = r["result"]["outputs"]["fitted"]
        else:
            log(f"fit to the photo not kept: {st.get('reason') or st.get('error')}")
        return st

    def _fill(self, inp, ctx, options, progress, cancel, log, scene, locked_ply, cam, mg, used):
        """Holes of moved views → painted → SHARP with tied depth → added. None when it cannot run."""
        from PIL import Image

        from ..fusion import DepthBuffer, add_fills, unseen_by_photo
        from . import painter

        if not painter.installed(ctx):
            self.fill_note = "skipped: needs the AI painter (Components: AI painter engine and Qwen-Image-Edit)"
            return None
        mode = options.get("mode", "auto")
        work = inp.work_dir / "locked" / "fill"
        work.mkdir(parents=True, exist_ok=True)
        pivot = pivot_depth(scene, cam)
        cams = sheet_cameras(cam, pivot, side=1024, views=FILL_VIEWS.get(mode, FILL_VIEWS["auto"]), photo=False)
        progress(0.46, f"looking from {len(cams)} moved positions")
        renders = render_views(ctx, locked_ply, scene, [c for _, c in cams], work / "render", work,
                               self.render_script, log, cpu_fallback=True)
        if renders is None:
            self.fill_note = "skipped: no renderer could draw the moved views (see job.log)"
            return None
        photo_buf = DepthBuffer(cam).fill(scene.means, scene.opacity > 0.1)
        todo = []
        for k, ((label, c), r) in enumerate(zip(cams, renders)):
            vc = view_camera(c)
            hole = unseen_by_photo(r["depth"], r["alpha"], vc, vc.c2w, photo_buf)
            for _ in range(2):                        # cover the fringe of stretched splats too
                g = hole.copy()
                g[1:] |= hole[:-1]
                g[:-1] |= hole[1:]
                g[:, 1:] |= hole[:, :-1]
                g[:, :-1] |= hole[:, 1:]
                hole = g
            if hole.mean() < MIN_HOLE:
                log(f"{label}: nothing to fill")
                continue
            rgb = r["rgb"].copy()
            rgb[hole] = 0
            base = work / f"v{k:02d}"
            Image.fromarray(rgb).save(f"{base}_holes.png")
            Image.fromarray(hole.astype(np.uint8) * 255).save(f"{base}_mask.png")
            np.savez(f"{base}_known.npz", depth=np.where(hole, np.inf, r["depth"]).astype(np.float32))
            todo.append({"k": k, "label": label, "cam": c, "hole": hole, "render": r["rgb"], "base": str(base)})
        if not todo:
            self.fill_note = "nothing to fill: no holes seen from the moved positions"
            return None
        log(f"painting the holes of {len(todo)} views with Qwen-Image-Edit")
        req = painter.request(ctx, f"{todo[0]['base']}_holes.png", str(work / "painted"),
                              [{"label": t["label"], "prompt": FILL_PROMPT, "image": f"{t['base']}_holes.png"}
                               for t in todo], 1.0 if mode != "fast" else 0.75, int(options.get("seed", 42)))
        painted = painter.paint(ctx, req, inp.work_dir / "worker", progress, cancel, log, 0.5, 0.75,
                                self.painter_script)
        for m in painter.MODELS:
            used(m)
        if len(painted) != len(todo):
            raise BackendError(f"the painter returned {len(painted)} pictures for {len(todo)} views")
        for t, p in zip(todo, painted):
            c = t["cam"]
            with Image.open(p) as im:
                g = np.asarray(im.convert("RGB").resize((c["width"], c["height"]), Image.LANCZOS))
            filled = t["render"].copy()
            filled[t["hole"]] = g[t["hole"]]           # everything visible stays exactly the render
            t["filled"] = filled
            Image.fromarray(filled).save(f"{t['base']}_filled.png")

        # SHARP on every filled view, its depth tied to the scene
        out = run_worker(ctx.runtimes.python(self.runtime_id), self.worker_script,
                         {"mode": "fill", "checkpoint": str(ctx.models.paths("sharp")["sharp_2572gikvuh.pt"]),
                          "moge_path": str(ctx.models.paths(mg)["model.pt"]),
                          "moge_version": "v3" if mg.startswith("moge-3") else "v2",
                          "views": [{"image": f"{t['base']}_filled.png", "mask": f"{t['base']}_mask.png",
                                     "known": f"{t['base']}_known.npz", "fx": t["cam"]["fx"], "fy": t["cam"]["fy"],
                                     "cx": t["cam"]["cx"], "cy": t["cam"]["cy"], "out": f"{t['base']}_fill.ply"}
                                    for t in todo]},
                         inp.work_dir / "worker", progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id),
                         log=log, progress_range=(0.75, 0.88), timeout_s=3600)
        fills = []
        for t, ply in zip(todo, out["result"]["outputs"]):
            sc = read_gaussian_ply(Path(ply))
            vc = view_camera(t["cam"])
            fills.append((sc, vc, vc.c2w, t["hole"], t["label"]))
        progress(0.9, "joining the new content")
        merged, added, report = add_fills(scene, cam, fills, log=log)
        for rep, st in zip(report, out["result"].get("stats") or []):
            rep["depth_tie"] = st

        # what each step did, in one picture
        steps = None
        try:
            mply = write_gaussian_ply(merged, work / "merged.ply", include_provenance=True)
            after = render_views(ctx, mply, merged, [t["cam"] for t in todo], work / "after", work,
                                 self.render_script, log)
            rows = []
            for t, a in zip(todo, after):
                shown = t["render"].copy()
                shown[t["hole"]] = (255, 0, 255)
                rows.append((t["label"], [shown, t["filled"], a["rgb"]]))
            steps = grid(rows, ["holes (magenta)", "painted", "result"], inp.work_dir / "export" / "fill_steps.png")
        except Exception as e:  # noqa: BLE001 - a diagnostic
            log(f"fill steps picture skipped: {e}")
        return merged, added, report, steps

    def export(self, result: BackendResult, destination: Path) -> list[Path]:
        out = super().export(result, destination)
        for p in (result.extra.get("files") or {}).values():
            if p and Path(p).exists():
                tgt = destination / Path(p).name
                if Path(p).resolve() != tgt.resolve():
                    shutil.copy2(p, tgt)
                out.append(tgt)
        return out
