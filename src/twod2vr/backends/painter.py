"""The AI painter of Real 3D: Qwen-Image-Edit-2511 (Alibaba Qwen, Apache-2.0, GGUF Q5) with the Lightning
LoRA (4 steps), in the 'gen-cu128' runtime. It paints only the holes of views of a photo's splat from
moved cameras (backends/locked.py); everything visible stays the render.

Two stages, each in its own process (workers/qwen_views_worker.py): "encode" (the 7B vision-language text
encoder) and "generate" (the 20B image transformer streamed block by block). Each memory setting runs in a
fresh process; the worker asks for a lighter one ('oom_retry') when one does not fit.
"""

from __future__ import annotations

from .base import BackendError, run_worker

RUNTIME = "gen-cu128"
SCRIPT = "qwen_views_worker.py"
MODELS = ("qwen-image-edit-2511-q5", "qwen-image-edit-2511-base", "qwen-edit-2511-lightning")


def installed(ctx) -> bool:
    return ctx.runtimes.is_installed(RUNTIME) and all(m in ctx.models.entries and ctx.models.is_installed(m)
                                                       for m in MODELS)


def request(ctx, image: str, output_dir: str, views: list[dict], megapixels: float, seed: int) -> dict:
    paths = {m: ctx.models.paths(m) for m in MODELS}
    return {"image": image, "output_dir": output_dir,
            "base_dir": str(ctx.models.model_dir("qwen-image-edit-2511-base")),
            "gguf": str(next(iter(paths["qwen-image-edit-2511-q5"].values()))),
            "lora_lightning": str(next(iter(paths["qwen-edit-2511-lightning"].values()))),
            "views": views, "angles_strength": 0.0, "candidates": 1, "megapixels": megapixels, "steps": 4,
            "seed": int(seed)}


def run_attempts(ctx, req: dict, work_dir, progress, cancel, log, progress_range, script: str | None = None) -> dict:
    """One stage in fresh processes until a memory setting fits (only an exiting process frees VRAM/RAM)."""
    from ..sysmem import memory_status

    for attempt in range(8):
        req["attempt"] = attempt
        st = memory_status()
        if st is not None and log:
            log(f"memory: {st.summary()}")
        try:
            return run_worker(ctx.runtimes.python(RUNTIME), script or SCRIPT, req, work_dir, progress, cancel,
                              env=ctx.runtimes.worker_env(RUNTIME), log=log, progress_range=progress_range,
                              timeout_s=4 * 3600)
        except BackendError as e:
            if e.code != "oom_retry":
                raise
            if log:
                log(str(e))
    raise BackendError("no memory setting worked", code="oom")


def paint(ctx, req: dict, work_dir, progress, cancel, log, lo: float, hi: float, script: str | None = None) -> list[str]:
    """Encode then generate; the painted pictures in the order of req['views']."""
    mid = lo + 0.2 * (hi - lo)
    run_attempts(ctx, {**req, "stage": "encode"}, work_dir, progress, cancel, log, (lo, mid), script)
    out = run_attempts(ctx, {**req, "stage": "generate"}, work_dir, progress, cancel, log, (mid, hi), script)
    return [v["path"] for v in out["result"]["views"] if v.get("generated")]
