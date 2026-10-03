"""Photo → generated views from other angles → one 3D scene.

Three generation engines:
  * Qwen-Image-Edit-2511 + Multiple-Angles LoRA (Alibaba Qwen / fal, Apache-2.0) - an image model that
    redraws the photo from a requested camera position; each view is a sharp ~1 MP image;
  * Wan 2.2 Fun 5B Control-Camera (Alibaba PAI, Apache-2.0) - a video model that films camera moves
    starting at the photo; key frames at the requested angles are used; keeps people intact;
  * Stable Virtual Camera (Stability AI, non-commercial) - multi-view diffusion; scenes and objects.
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
    "tri": "3 views: 45° to the left, 45° to the right and a high-angle shot from above, reconstructed with the "
           "photo by the multi-view engine (trained splat)",
    "capture": "360° photo capture: 45°, 90°, 135°, 180° and 270°, overhead and from below (Quality adds 225°, "
               "315° and raised diagonals)",
    "arc": "around the subject: 45° to each side, from above and from below",
    "orbit": "wide orbit around the main subject (±100°, above and below with Wan 2.2; 360° with Stable "
             "Virtual Camera)",
    "explore": "look around from where the photo was taken, with a little head movement (best for VR180)",
    "spiral": "small forward-facing spiral (most faithful, least new content)",
}
FRAMES = {"fast": 48, "auto": 80, "quality": 110}          # Stable Virtual Camera views
# Wan: (frames per shot (4k+1), sampling steps). Sharp fusion only uses a few key views per shot, so its
# shots can be shorter; a trained splat wants many frames. 360° capture orbits up to 180° in one shot.
WAN_SETTINGS = {"fusion": {"fast": (25, 25), "auto": (33, 30), "quality": (49, 40)},
                "train": {"fast": (49, 30), "auto": (81, 50), "quality": (81, 50)},
                "capture": {"fast": (33, 25), "auto": (49, 30), "quality": (49, 40)}}
FUSION_SEVA_VIEWS = 16
SEVA_MODELS = ("seva-1.1", "sd21-vae", "clip-vit-h-14")
WAN_MODELS = ("wan2.2-fun-5b-camera",)
QWEN_MODELS = ("qwen-image-edit-2511-q5", "qwen-image-edit-2511-base", "qwen-edit-2511-angles-lora",
               "qwen-edit-2511-lightning")
QWEN_LIGHTNING_8 = "qwen-edit-2511-lightning-8"   # optional: Quality mode samples 8 steps with it
FLASHWORLD_MODELS = ("flashworld", "wan2.2-ti2v-5b-base")
ENGINE_MODELS = {"qwen": QWEN_MODELS, "wan": WAN_MODELS, "seva": SEVA_MODELS, "flashworld": FLASHWORLD_MODELS}
ENGINE_NAMES = {"flashworld": "FlashWorld (3D Gaussians generated directly, Wan2.2-TI2V-5B)",
                "qwen": "Qwen-Image-Edit-2511 + Multiple-Angles LoRA", "wan": "Wan 2.2 Fun 5B Control-Camera",
                "seva": "Stable Virtual Camera"}
ENGINE_TRAJECTORIES = {"qwen": ("tri", "capture", "arc", "orbit"), "wan": tuple(TRAJECTORIES),
                       "seva": ("arc", "orbit", "explore", "spiral", "capture")}
ENGINE_TRAJECTORIES["wan"] = tuple(t for t in TRAJECTORIES if t != "tri")
# FlashWorld generates the whole 3D at once (no separate views): it serves "Real 3D from one photo"
ENGINE_TRAJECTORIES["flashworld"] = ("tri",)
# FlashWorld per quality mode: (swing either side in degrees, max splats)
FLASHWORLD_SETTINGS = {"fast": (25, 1_500_000), "auto": (30, 3_000_000), "quality": (35, 4_000_000)}

# Qwen views: (label, azimuth°, elevation°). Azimuth clockwise seen from above (90 = right side,
# 180 = back); elevation + = camera above. The LoRA knows 8 azimuths and -30/0/30/60° elevations.
_CAPTURE = [("45°", 45, 0), ("90°", 90, 0), ("135°", 135, 0), ("180°", 180, 0), ("270°", 270, 0),
            ("overhead", 0, 60), ("from below", 0, -30)]
QWEN_VIEWS = {
    "tri": [("45° left", 315, 0), ("45° right", 45, 0), ("high angle", 0, 30)],
    "capture": _CAPTURE,
    "capture_full": _CAPTURE + [("225°", 225, 0), ("315°", 315, 0), ("45° raised", 45, 30), ("135° raised", 135, 30),
                                ("225° raised", 225, 30), ("315° raised", 315, 30)],
    "arc": [("45°", 45, 0), ("315°", 315, 0), ("above", 0, 30), ("below", 0, -30), ("45° raised", 45, 30),
            ("315° raised", 315, 30)],
    "orbit": [(f"{a}°", a, 0) for a in (45, 90, 135, 180, 225, 270, 315)] + [("overhead", 0, 60),
                                                                                ("from below", 0, -30)],
}
QWEN_MEGAPIXELS = {"fast": 0.75, "auto": 1.0, "quality": 1.0}
TRI_SIDE = {"fast": 960, "auto": 1280, "quality": 1600}      # training resolution for photo + 3 views

# Candidates generated per view (the most rigid one - a pure camera move of the photo - is kept).
QWEN_CANDIDATES = {"fast": 1, "auto": 2, "quality": 3}
# A '3 views' angle whose best candidate scores above this gets RETRY_EXTRA more candidates (once).
RETRY_THRESHOLD = 0.45
RETRY_EXTRA = 2


def installed(ctx, engine: str) -> bool:
    return all(m in ctx.models.entries and ctx.models.is_installed(m) for m in ENGINE_MODELS[engine])


def generation_engine(ctx, preferred: str | None = None, trajectory: str | None = None) -> str | None:
    """The engine to use: the preferred one if installed (and able to do the trajectory), else the
    first installed of Qwen, FlashWorld (only for "tri"), Wan, Stable Virtual Camera. Qwen first: its
    views assembled with Depth Anything 3 gave the owner the most real 3D (rc36: FlashWorld less so)."""
    order = ["qwen", "flashworld", "wan", "seva"]
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    for e in order:
        if trajectory and trajectory not in ENGINE_TRAJECTORIES[e]:
            continue
        if installed(ctx, e):
            return e
    return None


def contact_sheet(photo, views: list[dict], selection: list[dict], out_path, thumb: int = 300):
    """One image to judge the generated views: the photo, then one row per angle with every candidate,
    its consistency score (lower = more like a pure camera move) and which one was used or dropped."""
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageOps

    sel = {s.get("view"): s for s in selection or []}
    gen = [v for v in views if v.get("generated")]
    if not gen:
        return None
    ncol = max(len(v.get("candidates") or [v["path"]]) for v in gen)
    label_h, pad = 34, 8
    W = pad + ncol * (thumb + pad)
    H = pad + (1 + len(gen)) * (thumb + label_h + pad)
    sheet = Image.new("RGB", (max(W, thumb + 2 * pad), H), (24, 26, 31))
    d = ImageDraw.Draw(sheet)

    def put(path, x, y, caption, colour=(230, 232, 238), border=None):
        try:
            with Image.open(path) as im:
                im = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (thumb, thumb))
        except OSError:
            return
        sheet.paste(im, (x, y + label_h))
        if border:
            d.rectangle([x - 3, y + label_h - 3, x + im.width + 2, y + label_h + im.height + 2], outline=border,
                        width=4)
        d.text((x, y + 8), caption, fill=colour)

    put(photo, pad, pad, "your photo (reference)")
    for r, v in enumerate(gen, start=1):
        s = sel.get(v.get("label"), {})
        y = pad + r * (thumb + label_h + pad)
        for c, path in enumerate(v.get("candidates") or [v["path"]]):
            name = Path(path).name
            score = (s.get("scores") or {}).get(name)
            chosen = s.get("chosen") == name or (not s and c == 0)
            dropped = chosen and s.get("dropped")
            tag = "DROPPED" if dropped else ("USED" if chosen else "")
            cap = f"{v.get('label', '')} · {name}" + (f" · score {score:.3f}" if score is not None else "") + \
                (f" · {tag}" if tag else "")
            put(path, pad + c * (thumb + pad), y, cap,
                border=(220, 70, 70) if dropped else ((70, 200, 110) if chosen else None))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return out_path


def target_angle(view: dict) -> float:
    """Camera change asked for a generated view, in degrees (for checking that it really moved)."""
    az = float(view.get("azimuth", 0)) % 360
    return max(min(az, 360 - az), abs(float(view.get("elevation", 0))))


def detailed(traj: str, mode: str) -> str:
    """360° capture in Quality mode adds the intermediate angles."""
    return "capture_full" if traj == "capture" and mode == "quality" else traj


class GenerativeSceneBackend(Backend):
    id = "generative_scene"
    display_name = "Generative 3D from one photo (Qwen-Image-Edit / Wan 2.2 / Stable Virtual Camera + fusion)"
    inputs = ("photo", "video:static_scene")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC, NOVEL_VIEW_COMPLETION})
    runtime_id = "gen-cu128"
    maturity = "experimental"
    upstream = ["qwen-image-edit", "videox-fun", "stable-virtual-camera", "vggt", "moge", "gsplat"]
    commercial_use = False  # VGGT-1B (reconstruction) and Stable Virtual Camera weights are non-commercial
    worker_script = "seva_worker.py"
    wan_script = "wan_worker.py"
    qwen_script = "qwen_views_worker.py"
    description = ("Generates the unseen sides of a photo with a camera-controlled diffusion model and trains a "
                   "real 3D Gaussian splat from the generated views. Invented content is labelled GENERATIVE.")

    def __init__(self):
        self.mv = MultiViewBackend()

    def model_ids(self, options: dict) -> list[str]:
        engine = options.get("engine") or getattr(self, "_engine", None) or "wan"
        return list(ENGINE_MODELS.get(engine, WAN_MODELS)) + self.mv.model_ids(options)

    def _pick(self, ctx, options: dict) -> str:
        pref = options.get("engine")
        return generation_engine(ctx, pref if pref != "auto" else None, options.get("trajectory")) or \
            (pref if pref in ENGINE_MODELS else "wan")

    def prepare(self, ctx: BackendContext, options: dict) -> None:
        self._engine = self._pick(ctx, options)
        super().prepare(ctx, {**options, "engine": self._engine})

    def min_vram_gb(self, options: dict) -> float | None:
        return 12

    def availability(self, ctx: BackendContext, hw, options: dict | None = None) -> Availability:
        options = dict(options or {})
        options["engine"] = self._pick(ctx, options)
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

    def _run_attempts(self, ctx, script, req, inp, progress, cancel, log, progress_range):
        """Each memory setting runs in a fresh process (only an exiting process reliably frees VRAM and
        RAM); the worker reports 'oom_retry' while a lighter setting is left and reuses finished work."""
        for attempt in range(8):
            req["attempt"] = attempt
            try:
                return run_worker(ctx.runtimes.python(self.runtime_id), script, req, inp.work_dir / "worker",
                                  progress, cancel, env=ctx.runtimes.worker_env(self.runtime_id), log=log,
                                  progress_range=progress_range, timeout_s=4 * 3600)
            except BackendError as e:
                if e.code != "oom_retry":
                    raise
                log(str(e))
        raise BackendError("no memory setting worked", code="oom")

    def run(self, inp: JobInput, ctx: BackendContext, options: dict, progress, cancel) -> BackendResult:
        traj = options.get("trajectory") or "capture"
        if traj not in TRAJECTORIES:
            traj = "capture"
        engine = getattr(self, "_engine", None) or self._pick(ctx, options)
        if traj not in ENGINE_TRAJECTORIES[engine]:
            raise BackendError(f"{ENGINE_NAMES[engine]} cannot do the '{traj}' path; choose 360° capture, Around "
                               "the subject or Wide orbit, or install Wan 2.2.", code="unsupported")
        if engine == "flashworld":
            return self._run_flashworld(inp, ctx, options, progress, cancel)
        mode = options.get("mode", "auto")
        assembly = options.get("assembly") or "fusion"
        if assembly not in ("fusion", "train"):
            assembly = "fusion"
        if traj == "tri":
            assembly = "train"   # photo + 3 views → the multi-view engine's trained splat
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        log = options.get("log") or (lambda m: None)
        out_dir = str(inp.work_dir / "generated_views")
        name, models = ENGINE_NAMES[engine], ENGINE_MODELS[engine]
        plan = detailed(traj, mode)
        log(f"generating views with {name}: {TRAJECTORIES[traj]}")
        for m in models:
            ctx.models.paths(m)  # clear error when missing
        if engine == "qwen":
            paths = {m: ctx.models.paths(m) for m in QWEN_MODELS}
            req = {"image": str(ref), "output_dir": out_dir,
                   "base_dir": str(ctx.models.model_dir("qwen-image-edit-2511-base")),
                   "gguf": str(next(iter(paths["qwen-image-edit-2511-q5"].values()))),
                   "lora_angles": str(next(iter(paths["qwen-edit-2511-angles-lora"].values()))),
                   "lora_lightning": str(next(iter(paths["qwen-edit-2511-lightning"].values()))),
                   "views": [{"label": lb, "azimuth": az, "elevation": el} for lb, az, el in QWEN_VIEWS[plan]],
                   "angles_strength": 0.9, "candidates": QWEN_CANDIDATES.get(mode, 1) if traj == "tri" else 1,
                   "distance": options.get("qwen_distance", "medium shot"),
                   "megapixels": QWEN_MEGAPIXELS.get(mode, 1.0), "steps": 4, "seed": int(options.get("seed", 42))}
            if mode == "quality" and QWEN_LIGHTNING_8 in ctx.models.entries and \
                    ctx.models.is_installed(QWEN_LIGHTNING_8):
                req.update(lora_lightning=str(next(iter(ctx.models.paths(QWEN_LIGHTNING_8).values()))), steps=8)
                models = tuple(m for m in models if m != "qwen-edit-2511-lightning") + (QWEN_LIGHTNING_8,)
                log("Quality: 8-step Lightning LoRA")
            self._run_attempts(ctx, self.qwen_script, {**req, "stage": "encode"}, inp, progress, cancel, log,
                               (0.0, 0.1))
            out = self._run_attempts(ctx, self.qwen_script, {**req, "stage": "generate"}, inp, progress,
                                     cancel, log, (0.1, 0.4))
            if traj == "tri" and mode != "fast" and options.get("retry", True):
                out = self._retry_weak_views(req, out, inp, ctx, options, progress, cancel, log)
        elif engine == "wan":
            frames, steps = WAN_SETTINGS["capture" if traj == "capture" else assembly].get(mode, (33, 30))
            req = {"image": str(ref), "output_dir": out_dir, "trajectory": plan, "frames": frames,
                   "steps": steps, "short_side": 704, "hfov_deg": options.get("hfov_deg"),
                   "seed": int(options.get("seed", 42)), "model_dir": str(ctx.models.model_dir(WAN_MODELS[0])),
                   "every": 0 if assembly == "fusion" else 2}
            out = self._run_attempts(ctx, self.wan_script, req, inp, progress, cancel, log, (0.0, 0.5))
        else:
            n = FRAMES.get(mode, 80)
            req = {"image": str(ref), "output_dir": out_dir, "trajectory": "orbit" if traj == "capture" else traj,
                   "num_frames": n, "steps": 50, "short_side": 576, "hfov_deg": options.get("hfov_deg"),
                   "seed": int(options.get("seed", 23)), "seva_dir": str(ctx.models.model_dir("seva-1.1")),
                   "vae_dir": str(ctx.models.model_dir("sd21-vae")),
                   "clip_path": str(ctx.models.paths("clip-vit-h-14")["open_clip_model.safetensors"])}
            out = self._run_attempts(ctx, self.worker_script, req, inp, progress, cancel, log, (0.0, 0.5))
        views = out["result"]["views"]
        if assembly == "fusion":
            views = self.fusion_views(views)
            log(f"sharp fusion of the photo and {len(views) - 1} key views")
        # The real photo is sampled more often than any single generated view during training.
        w_in = max(4.0, len(views) / 10)
        mv_views = [{"path": v["path"], "generated": bool(v["generated"]), "weight": 1.0 if v["generated"] else w_in,
                     **({"candidates": v["candidates"], "label": v.get("label", ""),
                         "target_deg": target_angle(v)} if len(v.get("candidates") or []) > 1 else {})}
                    for v in views]
        gen_models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                      for m in models]
        mv_opts = {**options, "max_side": 1024, "assembly": assembly}
        if traj == "tri":
            # photo + 3 views: train at the photo's detail, and with view-independent-ish colour (SH 1) -
            # with so few views higher SH degrees overfit into colour flicker when the head moves
            mv_opts.update(max_side=TRI_SIDE.get(mode, 1280), sh_degree=1)
        res = self.mv.reconstruct(
            mv_views, inp, ctx, mv_opts, progress, cancel,
            progress_range=(0.5, 0.97), extra_models=gen_models,
            extra_warnings=[f"Generative completion: {len(views) - 1} views were invented by {name} "
                            f"('{traj}'). Splats taken from those views are labelled GENERATIVE; they are "
                            "plausible, not measured."])
        res.backend = self.id
        try:
            sheet = contact_sheet(ref, views, res.extra.get("candidate_selection") or [],
                                  inp.work_dir / "export" / "generated_views_sheet.jpg")
            if sheet:
                res.extra["contact_sheet"] = str(sheet)
                log(f"generated views overview: {sheet}")
        except Exception as e:  # noqa: BLE001 - only an overview image
            log(f"could not draw the generated-views overview: {e}")
        res.extra.update({"trajectory": traj, "engine": engine, "assembly": res.extra.get("assembly") or assembly,
                          "generated_views": len(views) - 1,
                          "generation_vram_peak_mib": out["result"].get("vram_peak_mib")})
        res.worker_env = {"generation": out["env"], "reconstruction": res.worker_env}
        return res

    flashworld_script = "flashworld_worker.py"

    def _run_flashworld(self, inp, ctx, options, progress, cancel) -> BackendResult:
        """One photo -> 3D Gaussians generated directly by FlashWorld (runs in the reconstruction runtime,
        which has gsplat): no separate views to reconcile, so no ghosting between them."""
        import hashlib
        import json
        from pathlib import Path

        import numpy as np

        from ..scene import GENERATIVE, INFERRED, Camera, read_gaussian_ply, write_gaussian_ply, write_splat
        from .multiview import encode_turntable

        log = options.get("log") or (lambda m: None)
        mode = options.get("mode", "auto")
        swing, cap = FLASHWORLD_SETTINGS.get(mode, FLASHWORLD_SETTINGS["auto"])
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        for m in FLASHWORLD_MODELS:
            ctx.models.paths(m)  # clear error when missing
        base = ctx.models.model_dir("wan2.2-ti2v-5b-base")
        ckpt = next(iter(ctx.models.paths("flashworld").values()))
        rt = self.mv.runtime_id
        py, env = ctx.runtimes.python(rt), ctx.runtimes.worker_env(rt)
        prompt = str(options.get("prompt", ""))
        cache = ctx.models.model_dir("flashworld") / "text-cache"
        embeds = cache / f"umt5_{hashlib.sha1(prompt.encode()).hexdigest()[:12]}.pt"
        if not embeds.exists():
            log("encoding the text prompt once (UMT5-XXL); it is cached for later runs")
            run_worker(py, self.flashworld_script, {"stage": "encode", "base_dir": str(base), "prompt": prompt,
                                                    "embeds_path": str(embeds)},
                       inp.work_dir / "worker", progress, cancel, env=env, log=log, progress_range=(0.0, 0.1),
                       timeout_s=3600)
        out_dir = inp.work_dir / "flashworld_out"
        req = {"stage": "generate", "image": str(ref), "output_dir": str(out_dir), "base_dir": str(base),
               "ckpt": str(ckpt), "embeds_path": str(embeds), "frames": 24, "swing_deg": swing,
               "rise_deg": 6, "max_gaussians": cap, "min_opacity": 0.01,
               "moge_path": str(ctx.models.paths("moge-2-vitl-normal")["model.pt"])
               if "moge-2-vitl-normal" in ctx.models.entries and ctx.models.is_installed("moge-2-vitl-normal")
               else None}
        log(f"generating the 3D scene directly with {ENGINE_NAMES['flashworld']}")
        out = run_worker(py, self.flashworld_script, req, inp.work_dir / "worker", progress, cancel, env=env,
                         log=log, progress_range=(0.1, 0.97), timeout_s=4 * 3600)
        res, files = out["result"], out["result"]["outputs"]
        scene = read_gaussian_ply(Path(files["ply"]))
        cams = json.loads(Path(files["cameras"]).read_text()) if Path(files["cameras"]).exists() else []
        scene.cameras = [Camera.from_dict(c) for c in cams]
        scene.metric_scale = bool(res.get("metric"))
        # splats the photo shows are INFERRED (its colours, generated depth); the rest is GENERATIVE
        prov = np.full(len(scene), GENERATIVE, np.uint8)
        if cams:
            c = cams[0]
            w2c = np.linalg.inv(np.asarray(c["c2w"], float))
            pc = scene.means @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = c["fx"] * pc[:, 0] / z + c["cx"]
                v = c["fy"] * pc[:, 1] / z + c["cy"]
            prov[(z > 0) & (u >= 0) & (u < c["width"]) & (v >= 0) & (v < c["height"])] = INFERRED
        scene.provenance = prov
        exp = inp.work_dir / "export"
        ply = write_gaussian_ply(scene, exp / "scene.ply")
        splat = write_splat(scene, exp / "scene.splat")
        warnings = ["Generative 3D: FlashWorld generated the scene from your photo. Splats your photo shows are "
                    "INFERRED (its colours, generated depth); everything else is GENERATIVE - plausible, "
                    "not measured."]
        turntable = None
        if files.get("turntable_frames"):
            turntable, err = encode_turntable(Path(files["turntable_frames"]), exp / "turntable.mp4")
            if err:
                warnings.append(f"Turntable preview not encoded: {err}")
        models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                  for m in FLASHWORLD_MODELS]
        n_inf = int((prov == INFERRED).sum())
        return BackendResult(
            backend=self.id, scene_ply=ply, splat=splat, obj=None, cameras=cams, metric_scale=scene.metric_scale,
            provenance_note=(f"FlashWorld generated {len(scene):,} splats directly in 3D: {n_inf:,} seen in your "
                             f"photo (INFERRED), {len(scene) - n_inf:,} invented (GENERATIVE)."),
            vram_peak_mib=res.get("vram_peak_mib"), worker_env={"reconstruction": out["env"]},
            models_used=models, warnings=warnings,
            extra={"engine": "flashworld", "trajectory": "tri", "assembly": "generated", "gaussians": len(scene),
                   "subject_distance_m": res.get("subject_distance_m"), "hfov_deg": res.get("hfov_deg"),
                   "views": res.get("views"), "size": res.get("size"),
                   "turntable": str(turntable) if turntable else None})

    def _retry_weak_views(self, req, out, inp, ctx, options, progress, cancel, log):
        """Score the candidates now; for every angle whose best candidate still does not look like a pure
        camera move of the photo, generate RETRY_EXTRA more and keep the result (the final selection runs
        again inside the reconstruction)."""
        views = out["result"]["views"]
        mv_views = [{"path": v["path"], "generated": bool(v["generated"]),
                     **({"candidates": v["candidates"], "label": v.get("label", ""), "target_deg": target_angle(v)}
                        if v.get("generated") else {})} for v in views]
        try:
            sel = self.mv.score(mv_views, inp, ctx, {**options, "max_side": 1024}, progress, cancel, (0.4, 0.45))
        except BackendError as e:
            log(f"could not score the generated views ({e}); keeping them as they are")
            return out
        best = {s["view"]: min((s.get("scores") or {"": 9.0}).values()) for s in sel}
        weak = [lb for lb, sc in best.items() if sc > float(options.get("retry_threshold", RETRY_THRESHOLD))]
        log("view consistency: " + ", ".join(f"{lb} {sc:.3f}" for lb, sc in best.items()))
        if not weak:
            return out
        log(f"generating {RETRY_EXTRA} more candidates for: {', '.join(weak)}")
        n = int(req.get("candidates", 1))
        req2 = {**req, "stage": "generate",
                "views": [dict(v, candidates=n + RETRY_EXTRA) if v["label"] in weak else v for v in req["views"]]}
        return self._run_attempts(ctx, self.qwen_script, req2, inp, progress, cancel, log, (0.45, 0.5))

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
