"""Photo -> generated views from other angles -> one trained 3D Gaussian splat.

Two generation engines:
  * Qwen-Image-Edit-2511 + Multiple-Angles LoRA (Alibaba Qwen / fal, Apache-2.0) - an image model that
    redraws the photo from a requested camera position; each view is a sharp ~1 MP image. Real 3D draws
    them ring by ring around the subject, each view conditioned on its accepted neighbour;
  * Wan 2.2 Fun 5B Control-Camera (Alibaba PAI, Apache-2.0) - a video model that films camera moves
    starting at the photo (used to explore the surroundings for VR180, or when Qwen is not installed).
The views and the photo are then reconstructed by the multi-view engine (VGGT cameras, MoGe-2 scale and
depth priors, gsplat training). Photo pixels are INFERRED, the rest GENERATIVE.
"""

from __future__ import annotations

from .base import (NOVEL_VIEW_COMPLETION, SCENE_STATIC, SINGLE_VIEW, Availability, Backend, BackendContext,
                   BackendError, BackendResult, Estimate, JobInput, run_worker)
from .multiview import MultiViewBackend

TRAJECTORIES = {
    "tri": "Real 3D: rings of views around the subject (45° each side and from above, then 90°, 135° and the "
           "back; Fast stops at 90°), each drawn from the photo and its neighbouring view",
    "capture": "360° capture: 45°, 90°, 135°, 180° and 270°, overhead and from below (Quality adds 225°, 315° "
               "and raised diagonals)",
    "explore": "explore the surroundings from where the photo was taken, with a little head movement "
               "(Wan 2.2; best for VR180)",
}
# Wan: (frames per shot (4k+1), sampling steps); 360° capture orbits up to 180° in one shot.
WAN_SETTINGS = {"train": {"fast": (49, 30), "auto": (81, 50), "quality": (81, 50)},
                "capture": {"fast": (33, 25), "auto": (49, 30), "quality": (49, 40)}}
WAN_MODELS = ("wan2.2-fun-5b-camera",)
QWEN_MODELS = ("qwen-image-edit-2511-q5", "qwen-image-edit-2511-base", "qwen-edit-2511-angles-lora",
               "qwen-edit-2511-lightning")
ENGINE_MODELS = {"qwen": QWEN_MODELS, "wan": WAN_MODELS}
ENGINE_NAMES = {"qwen": "Qwen-Image-Edit-2511 + Multiple-Angles LoRA", "wan": "Wan 2.2 Fun 5B Control-Camera"}
ENGINE_TRAJECTORIES = {"qwen": ("tri", "capture"), "wan": ("capture", "explore")}

# Qwen views: (label, azimuth°, elevation°). Azimuth clockwise seen from above (90 = right side,
# 180 = back); elevation + = camera above. The LoRA knows 8 azimuths and -30/0/30/60° elevations.
_CAPTURE = [("45°", 45, 0), ("90°", 90, 0), ("135°", 135, 0), ("180°", 180, 0), ("270°", 270, 0),
            ("overhead", 0, 60), ("from below", 0, -30)]
QWEN_VIEWS = {
    "tri": [("45° left", 315, 0), ("45° right", 45, 0), ("high angle", 0, 30)],
    "capture": _CAPTURE,
    "capture_full": _CAPTURE + [("225°", 225, 0), ("315°", 315, 0), ("45° raised", 45, 30), ("135° raised", 135, 30),
                                ("225° raised", 225, 30), ("315° raised", 315, 30)],
}
QWEN_MEGAPIXELS = {"fast": 0.75, "auto": 1.0, "quality": 1.0}

# Candidates generated per view (the most rigid one - a pure camera move of the photo - is kept).
QWEN_CANDIDATES = {"fast": 1, "auto": 2, "quality": 3}
# Real 3D training (photo + views), per mode: the photo at its own detail, the AI views at their native
# size (~1 MP, enlarging adds nothing), more steps for the larger photo. SH degree 1: with a handful of
# views, higher degrees memorise the views' small disagreements as view-dependent colour, which shows
# as texture noise (docs/research/from-rc19.md).
TRI_PHOTO_SIDE = {"fast": 1280, "auto": 1600, "quality": 2048}
TRI_VIEW_SIDE = 1328
TRI_STEPS = {"fast": 6000, "auto": 12000, "quality": 20000}
TRI_SH_DEGREE = 1


def installed(ctx, engine: str) -> bool:
    return all(m in ctx.models.entries and ctx.models.is_installed(m) for m in ENGINE_MODELS[engine])


def generation_engine(ctx, preferred: str | None = None, trajectory: str | None = None) -> str | None:
    """The engine to use: the preferred one if installed (and able to do the trajectory), else the
    first installed of Qwen, Wan."""
    order = ["qwen", "wan"]
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    for e in order:
        if trajectory and trajectory not in ENGINE_TRAJECTORIES[e]:
            continue
        if installed(ctx, e):
            return e
    return None


# Real 3D views drawn in rings around the subject (rc42). Each ring is generated after the previous one
# and every view in it is drawn from the photo AND the neighbouring view already accepted ("refs"), so
# clothes, hair and proportions stay consistent all the way round, and the back finally exists (rc19 never
# looked further than 45°). (label, azimuth°, elevation°, reference labels: the first one is also the view
# it is judged against when choosing candidates).
TRI_RINGS = [
    [("45° left", 315, 0, ()), ("45° right", 45, 0, ()), ("high angle", 0, 30, ())],
    [("90° left", 270, 0, ("45° left",)), ("90° right", 90, 0, ("45° right",))],
    [("135° left", 225, 0, ("90° left",)), ("135° right", 135, 0, ("90° right",))],
    [("back", 180, 0, ("135° right", "135° left"))],
]
TRI_RING_COUNT = {"fast": 2, "auto": 4, "quality": 4}
# Optional complete subject (TRELLIS.2 shape, rc45): runtime + models; Fast uses the 512 pipeline
SUBJECT_RUNTIME = "subject3d-cu128"
SUBJECT_MODELS = ("trellis2-4b-shape", "trellis-image-large-ss-dec", "trellis2-image-encoders")
SUBJECT_PIPELINE = {"fast": "512", "auto": "1024_cascade", "quality": "1024_cascade"}


MOGE3 = "moge-3-vitl"          # rc48: the photo's geometry by MoGe-3 when installed (MoGe-2 otherwise)


def moge3_installed(ctx) -> bool:
    return MOGE3 in ctx.models.entries and ctx.models.is_installed(MOGE3) and ctx.runtimes.is_installed("photo-cu128")


def subject_installed(ctx) -> bool:
    return ctx.runtimes.is_installed(SUBJECT_RUNTIME) and all(
        m in ctx.models.entries and ctx.models.is_installed(m) for m in SUBJECT_MODELS)


def angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Degrees between two camera directions given as (azimuth, elevation) around the subject."""
    import math

    def vec(az, el):
        az, el = math.radians(az), math.radians(el)
        return (math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az))

    u, v = vec(*a), vec(*b)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(u, v))))))


def target_angle(view: dict) -> float:
    """Camera change asked for a generated view, in degrees (for checking that it really moved)."""
    az = float(view.get("azimuth", 0)) % 360
    return max(min(az, 360 - az), abs(float(view.get("elevation", 0))))


def detailed(traj: str, mode: str) -> str:
    """360° capture in Quality mode adds the intermediate angles."""
    return "capture_full" if traj == "capture" and mode == "quality" else traj


class GenerativeSceneBackend(Backend):
    id = "generative_scene"
    display_name = "Generative 3D from one photo (Qwen-Image-Edit / Wan 2.2 + multi-view)"
    inputs = ("photo", "video:static_scene")
    capabilities = frozenset({SINGLE_VIEW, SCENE_STATIC, NOVEL_VIEW_COMPLETION})
    runtime_id = "gen-cu128"
    maturity = "experimental"
    upstream = ["qwen-image-edit", "videox-fun", "vggt", "moge", "gsplat"]
    commercial_use = False  # VGGT-1B (reconstruction) weights are non-commercial
    wan_script = "wan_worker.py"
    qwen_script = "qwen_views_worker.py"
    subject_script = "trellis_subject_worker.py"
    geometry_script = "moge_worker.py"
    description = ("Generates the unseen sides of a photo with a camera-controlled diffusion model and trains a "
                   "real 3D Gaussian splat from the generated views. Invented content is labelled GENERATIVE.")

    def __init__(self):
        self.mv = MultiViewBackend()

    def model_ids(self, options: dict) -> list[str]:
        engine = options.get("engine") or getattr(self, "_engine", None) or "qwen"
        return list(ENGINE_MODELS.get(engine, QWEN_MODELS)) + self.mv.model_ids(options)

    def _pick(self, ctx, options: dict) -> str:
        pref = options.get("engine")
        pref = pref if pref != "auto" else None
        traj = options.get("trajectory")
        # Real 3D without Qwen-Image-Edit: Wan 2.2's 360° capture instead (see run())
        return generation_engine(ctx, pref, traj) or (generation_engine(ctx, pref, "capture") if traj == "tri"
                                                      else None) or (pref if pref in ENGINE_MODELS else "qwen")

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
        n = {"fast": 5, "auto": 16, "quality": 24}.get(options.get("mode", "auto"), 16)   # Qwen images
        return Estimate(seconds=180 + n * 25 + 600, vram_gb=12, disk_gb=1.0,
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
        traj = options.get("trajectory") or "tri"
        if traj not in TRAJECTORIES:
            traj = "tri"
        engine = getattr(self, "_engine", None) or self._pick(ctx, options)
        log = options.get("log") or (lambda m: None)
        if traj == "tri" and engine == "wan":
            traj = "capture"
            log("Qwen-Image-Edit is not installed: Real 3D uses Wan 2.2's 360° capture instead")
        if traj not in ENGINE_TRAJECTORIES[engine]:
            alt = "Qwen-Image-Edit" if engine == "wan" else "Wan 2.2"
            raise BackendError(f"{ENGINE_NAMES[engine]} cannot do the '{traj}' path; choose another path or "
                               f"install {alt}.", code="unsupported")
        mode = options.get("mode", "auto")
        ref = inp.frames[min(int(options.get("reference_frame_index", 0)), len(inp.frames) - 1)]
        out_dir = str(inp.work_dir / "generated_views")
        name, models = ENGINE_NAMES[engine], ENGINE_MODELS[engine]
        plan = detailed(traj, mode)
        log(f"generating views with {name}: {TRAJECTORIES[traj]}")
        for m in models:
            ctx.models.paths(m)  # clear error when missing
        subject = None
        if engine == "qwen" and traj == "tri":
            out = None
            if options.get("complete_subject", True) and subject_installed(ctx):
                subject = self._complete_subject(inp, ctx, options, progress, cancel, log, ref, mode)
            chain = self._qwen_rings(inp, ctx, options, progress, cancel, log, ref, out_dir, mode,
                                     start=0.08 if subject else 0.0)
        elif engine == "qwen":
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
            self._run_attempts(ctx, self.qwen_script, {**req, "stage": "encode"}, inp, progress, cancel, log,
                               (0.0, 0.1))
            out = self._run_attempts(ctx, self.qwen_script, {**req, "stage": "generate"}, inp, progress,
                                     cancel, log, (0.1, 0.5))
        elif engine == "wan":
            frames, steps = WAN_SETTINGS["capture" if traj == "capture" else "train"].get(mode, (33, 30))
            req = {"image": str(ref), "output_dir": out_dir, "trajectory": plan, "frames": frames,
                   "steps": steps, "short_side": 704, "hfov_deg": options.get("hfov_deg"),
                   "seed": int(options.get("seed", 42)), "model_dir": str(ctx.models.model_dir(WAN_MODELS[0])),
                   "every": 2}
            out = self._run_attempts(ctx, self.wan_script, req, inp, progress, cancel, log, (0.0, 0.5))
        views = out["result"]["views"] if out is not None else chain["views"]
        # The real photo is sampled more often than any single generated view during training.
        w_in = max(4.0, len(views) / 10)
        mv_views = [{"path": v["path"], "generated": bool(v["generated"]), "weight": 1.0 if v["generated"] else w_in,
                     # the photo's real lens (EXIF of the original file; the frame is a PNG copy without it)
                     **({"fov_x": options["hfov_deg"]} if not v["generated"] and options.get("hfov_deg") else {}),
                     **({"candidates": v["candidates"], "label": v.get("label", ""),
                         "target_deg": target_angle(v)} if len(v.get("candidates") or []) > 1 else {})}
                    for v in views]
        gen_models = [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                      for m in models]
        mv_opts = {**options, "max_side": 1024}
        if traj == "tri":
            for v in mv_views:
                v["max_side"] = TRI_VIEW_SIDE if v["generated"] else TRI_PHOTO_SIDE.get(mode, 1600)
            mv_opts.update(sh_degree=TRI_SH_DEGREE, train_steps=TRI_STEPS.get(mode, 12000))
            if subject:
                mv_opts.update(subject_mesh=subject["subject_mesh"], subject_mask=subject["subject_mask"])
                gen_models += subject["models"]
            geo = self._photo_geometry(inp, ctx, options, progress, cancel, log, ref, mode) \
                if moge3_installed(ctx) else None
            if geo:
                mv_opts["photo_geometry"] = geo
                gen_models.append({"id": MOGE3, **{k: ctx.models.status(MOGE3)[k]
                                                   for k in ("revision", "license", "hash_status")}})
        res = self.mv.reconstruct(
            mv_views, inp, ctx, mv_opts, progress, cancel,
            progress_range=(0.5, 0.97), extra_models=gen_models,
            extra_warnings=[f"Generative completion: {len(views) - 1} views were invented by {name} "
                            f"('{traj}'). Splats taken from those views are labelled GENERATIVE; they are "
                            "plausible, not measured."])
        res.backend = self.id
        res.extra.update({"trajectory": traj, "engine": engine, "assembly": "train",
                          "generated_views": len(views) - 1,
                          "generation_vram_peak_mib": (out["result"] if out is not None else chain).get("vram_peak_mib")})
        if out is None:
            res.extra["candidate_selection"] = chain["selection"]
        res.worker_env = {"generation": (out if out is not None else chain)["env"], "reconstruction": res.worker_env}
        return res

    def _photo_geometry(self, inp, ctx, options, progress, cancel, log, ref, mode):
        """The photo's metric geometry by MoGe-3 (photo runtime), at its training size and with its real
        field of view; None if it fails (MoGe-2 inside the multi-view worker is the fallback)."""
        out = inp.work_dir / "photo_geometry"
        out.mkdir(parents=True, exist_ok=True)
        item = {"path": str(ref), "out": str(out / "photo.npz")}
        if options.get("hfov_deg"):
            item["fov_x"] = float(options["hfov_deg"])
        req = {"model_path": str(ctx.models.paths(MOGE3)["model.pt"]), "version": "v3", "refine_steps": 3,
               "max_side": TRI_PHOTO_SIDE.get(mode, 1600), "fp16": True, "images": [item]}
        log("the photo's geometry: MoGe-3")
        try:
            run_worker(ctx.runtimes.python("photo-cu128"), self.geometry_script, req, inp.work_dir / "worker", progress,
                       cancel, env=ctx.runtimes.worker_env("photo-cu128"), log=log, progress_range=(0.5, 0.52),
                       timeout_s=1800)
        except BackendError as e:
            log(f"MoGe-3 unavailable ({e}); MoGe-2 is used")
            return None
        return item["out"]

    def _complete_subject(self, inp, ctx, options, progress, cancel, log, ref, mode):
        """TRELLIS.2 shape of the whole subject (back included), or None if it fails (Real 3D goes on
        without it). Its mesh guides the depth of the AI views in the multi-view worker."""
        p = {m: ctx.models.paths(m) for m in SUBJECT_MODELS}
        shape_dir = ctx.models.model_dir("trellis2-4b-shape")
        req = {"image": str(ref), "output_dir": str(inp.work_dir / "complete_subject"),
               "model_dir": str(shape_dir), "image_large_dir": str(ctx.models.model_dir("trellis-image-large-ss-dec")),
               "dinov3_zip": str(p["trellis2-image-encoders"]["dinov3.zip"]),
               "rmbg_zip": str(p["trellis2-image-encoders"]["RMBG-2.0.zip"]),
               "pipeline_type": SUBJECT_PIPELINE.get(mode, "1024_cascade"), "seed": int(options.get("seed", 42))}
        log(f"complete subject: TRELLIS.2 ({req['pipeline_type']})")
        try:
            out = run_worker(ctx.runtimes.python(SUBJECT_RUNTIME), self.subject_script, req, inp.work_dir / "worker",
                             progress, cancel, env=ctx.runtimes.worker_env(SUBJECT_RUNTIME), log=log,
                             progress_range=(0.0, 0.08), timeout_s=3600)
        except BackendError as e:
            log(f"complete subject unavailable ({e}); Real 3D continues without it")
            return None
        files = out["result"]["outputs"]
        return {"subject_mesh": files["mesh"], "subject_mask": files["mask"],
                "models": [{"id": m, **{k: ctx.models.status(m)[k] for k in ("revision", "license", "hash_status")}}
                           for m in SUBJECT_MODELS]}

    def _qwen_rings(self, inp, ctx, options, progress, cancel, log, ref, out_dir, mode, start=0.0):
        """Real 3D views, ring by ring (TRI_RINGS): generate a ring (each view from the photo + its
        neighbouring accepted view), choose the most consistent candidate of each view (judged against
        that neighbour, VGGT score-only pass), then the next ring uses those choices as references."""
        paths = {m: ctx.models.paths(m) for m in QWEN_MODELS}
        n_cand = QWEN_CANDIDATES.get(mode, 1)
        base = {"image": str(ref), "output_dir": out_dir,
                "base_dir": str(ctx.models.model_dir("qwen-image-edit-2511-base")),
                "gguf": str(next(iter(paths["qwen-image-edit-2511-q5"].values()))),
                "lora_angles": str(next(iter(paths["qwen-edit-2511-angles-lora"].values()))),
                "lora_lightning": str(next(iter(paths["qwen-edit-2511-lightning"].values()))),
                "angles_strength": 0.9, "candidates": n_cand,
                "distance": options.get("qwen_distance", "medium shot"),
                "megapixels": QWEN_MEGAPIXELS.get(mode, 1.0), "steps": 4, "seed": int(options.get("seed", 42))}
        rings = TRI_RINGS[:TRI_RING_COUNT.get(mode, 4)]
        chosen: dict[str, dict] = {}            # label -> accepted view (path, azimuth, elevation)
        selection, peak, index = [], None, 0
        span = (0.5 - start) / len(rings)
        for r, ring in enumerate(rings):
            lo = start + r * span
            views = []
            for label, az, el, refs in ring:
                views.append({"label": label, "azimuth": az, "elevation": el, "index": index,
                              "refs": [{"path": chosen[x]["path"], "label": x} for x in refs if x in chosen]})
                index += 1
            req = {**base, "views": views}
            log(f"ring {r + 1}/{len(rings)}: " + ", ".join(
                v["label"] + (f" (with {', '.join(x['label'] for x in v['refs'])})" if v["refs"] else "")
                for v in views))
            self._run_attempts(ctx, self.qwen_script, {**req, "stage": "encode"}, inp, progress, cancel, log,
                               (lo, lo + span * 0.2))
            out = self._run_attempts(ctx, self.qwen_script, {**req, "stage": "generate"}, inp, progress, cancel,
                                     log, (lo + span * 0.2, lo + span * 0.8))
            peak = max(filter(None, [peak, out["result"].get("vram_peak_mib")]), default=None)
            gen = [v for v in out["result"]["views"] if v.get("generated")]
            picks = {v["label"]: v["path"] for v in gen}
            if n_cand > 1:
                order = [ref]
                items = [{"path": str(ref), "generated": False}]
                for lb, v in chosen.items():
                    items.append({"path": v["path"], "generated": True, "label": lb})
                    order.append(lb)
                for v, (label, az, el, refs) in zip(gen, ring):
                    anchor = refs[0] if refs and refs[0] in chosen else None
                    items.append({"path": v["path"], "generated": True, "label": label,
                                  "candidates": v["candidates"], "max_side": TRI_VIEW_SIDE,
                                  "target_deg": target_angle({"azimuth": az, "elevation": el}),
                                  **({"anchor": order.index(anchor),
                                      "anchor_deg": angle_between((az, el), (chosen[anchor]["azimuth"],
                                                                             chosen[anchor]["elevation"]))}
                                     if anchor else {})})
                try:
                    sel = self.mv.score(items, inp, ctx, options, progress, cancel, (lo + span * 0.8, lo + span))
                except BackendError as e:
                    log(f"could not score ring {r + 1} ({e}); keeping the first candidates")
                    sel = []
                for s in sel:
                    if s.get("path"):
                        picks[s["view"]] = s["path"]
                selection += sel
            for v, (label, az, el, _) in zip(gen, ring):
                chosen[label] = {"path": picks.get(label, v["path"]), "azimuth": az, "elevation": el}
        views = [{"path": str(ref), "generated": False}] + [
            {"path": v["path"], "generated": True, "key": True, "label": lb, "azimuth": v["azimuth"],
             "elevation": v["elevation"]} for lb, v in chosen.items()]
        return {"views": views, "selection": selection, "vram_peak_mib": peak, "env": out["env"]}
