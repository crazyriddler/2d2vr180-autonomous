"""Client for workers/render_worker.py (gsplat, true 3DGS compositing).

Frames are rendered in chunks so a long VR180 video never needs more than a
few hundred MB of temporary view files. Any failure is raised as
BackendError; callers fall back to the CPU reference renderer and say so.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .backends.base import BackendError, run_worker
from .scene import GaussianScene
from .vr180 import StereoFrame, StereoOptions, assemble_gpu, finalize_stereo, gpu_views, plan_stereo

# The video runtime ships gsplat's prebuilt CUDA wheel; the photo runtime's gsplat
# is the pure-Python PyPI build that JIT-compiles (needs MSVC + CUDA toolkit),
# so it is only a fallback.
RUNTIMES = ("recon3d-cu124", "photo-cu128")


def pick_runtime(ctx) -> str | None:
    return next((r for r in RUNTIMES if ctx.runtimes.is_installed(r)), None)


def gpu_renderer_available(ctx, hw) -> tuple[bool, str]:
    if hw is None or not hw.gpus:
        return False, "no NVIDIA GPU detected"
    if pick_runtime(ctx) is None:
        return False, "no runtime with gsplat is installed"
    return True, ""


class GpuSplatRenderer:
    def __init__(self, ctx, work_dir: Path, log: Callable[[str], None] | None = None, chunk_frames: int = 16):
        self.ctx = ctx
        self.work_dir = Path(work_dir)
        self.log = log
        self.chunk = chunk_frames

    def frames(self, scene: GaussianScene, scene_ply: Path, opts: StereoOptions, heads: list[np.ndarray],
               progress: Callable[[float, str], None], cancel: Callable[[], bool]) -> Iterator[StereoFrame]:
        n = len(heads)
        for c0 in range(0, n, self.chunk):
            plans = [plan_stereo(scene, opts, h) for h in heads[c0:c0 + self.chunk]]
            views, counts = [], []
            for p in plans:
                v = gpu_views(p)
                counts.append(len(v))
                views += v
            out_dir = self.work_dir / "gpu_views"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            req = {"ply": str(scene_ply), "out_dir": str(out_dir),
                   "views": [{**v, "c2w": np.asarray(v["c2w"]).tolist()} for v in views]}
            lo, hi = c0 / n, min(1.0, (c0 + len(plans)) / n)
            rt = pick_runtime(self.ctx)
            run_worker(self.ctx.runtimes.python(rt), "render_worker.py", req, self.work_dir / "worker",
                       lambda v, m: progress(lo + (hi - lo) * v, m), cancel,
                       env=self.ctx.runtimes.worker_env(rt), log=self.log)
            k = 0
            for p, cnt in zip(plans, counts):
                rendered = []
                for i in range(k, k + cnt):
                    f = out_dir / f"view_{i:06d}.npz"
                    if not f.exists():
                        raise BackendError(f"GPU renderer did not produce {f.name}", code="worker_crash")
                    with np.load(f) as d:
                        rendered.append({"rgb": d["rgb"], "alpha": d["alpha"], "depth": d["depth"]})
                k += cnt
                left, right = assemble_gpu(p, rendered)
                yield finalize_stereo(scene, opts, p, left, right, renderer="gpu-gsplat (3DGS alpha compositing)")
            shutil.rmtree(out_dir, ignore_errors=True)
