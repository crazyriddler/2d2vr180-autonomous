"""AI hole filling (LaMa) for stereo disocclusions.

Holes are the pixels that the renderer could not see from one eye (behind
foreground objects). Without LaMa they are filled by propagating the background
colour sideways; with LaMa (Apache-2.0, ~200 MB, runs in the generative engine)
they are inpainted with plausible texture. Either way they stay labelled as
interpolated in the coverage masks. The region outside the reconstructed field
of view is never touched here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RUNTIME = "gen-cu128"
WORKER = "inpaint_worker.py"
MODEL = "big-lama"


def lama_available(ctx) -> tuple[bool, str]:
    if not ctx.runtimes.is_installed(RUNTIME):
        return False, f"runtime '{RUNTIME}' not installed"
    if MODEL not in ctx.models.entries or not ctx.models.is_installed(MODEL):
        return False, f"model '{MODEL}' not installed"
    return True, ""


def hole_mask(r, grow: int = 2) -> np.ndarray:
    m = r.hole_filled.copy() if r.hole_filled is not None else np.zeros(r.covered.shape, bool)
    for _ in range(grow):  # dilate so seams at the hole border are blended too
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    known = r.covered | r.filled | (r.hole_filled if r.hole_filled is not None else False)
    return m & known


def lama_fill_frames(frames: list, ctx, work_dir: Path, progress, cancel, log=None) -> int:
    """Inpaint the disocclusion holes of each StereoFrame's eyes in place (both eyes).
    Returns the number of eye images changed."""
    from PIL import Image

    from .backends.base import run_worker
    from .vr180 import compose

    work_dir.mkdir(parents=True, exist_ok=True)
    items, eyes = [], []
    for i, fr in enumerate(frames):
        for side in ("left", "right"):
            r = getattr(fr, side)
            m = hole_mask(r)
            if not m.any():
                continue
            base = work_dir / f"f{i:05d}_{side}"
            Image.fromarray(r.rgb).save(base.with_suffix(".png"))
            Image.fromarray(m.astype(np.uint8) * 255).save(f"{base}_mask.png")
            items.append({"image": str(base.with_suffix(".png")), "mask": f"{base}_mask.png", "out": f"{base}_out.png"})
            eyes.append((fr, side))
    if not items:
        return 0
    run_worker(ctx.runtimes.python(RUNTIME), WORKER,
               {"model": str(ctx.models.paths(MODEL)["big-lama.pt"]), "items": items}, work_dir, progress, cancel,
               env=ctx.runtimes.worker_env(RUNTIME), log=log)
    for it, (fr, side) in zip(items, eyes):
        getattr(fr, side).rgb = np.asarray(Image.open(it["out"]).convert("RGB")).copy()
    for fr in frames:
        layout = fr.metadata.get("layout", "sbs")
        fr.image = compose(fr.left.rgb, fr.right.rgb, layout)
        fr.metadata["hole_filling"] = "LaMa AI inpainting of disocclusions (generated texture, not observed)"
    return len(items)
