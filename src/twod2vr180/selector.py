"""Pipeline selector: input analysis + hardware + installed runtimes/models +
licence profile + user mode (auto / quality / fast) → backend choice.

Every rejected candidate is returned with its reason so the UI and
run_report.json can explain the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import all_backends
from .backends.base import Backend, BackendContext, JobInput
from .hardware import HardwareReport

MODES = ("auto", "quality", "fast")

# Preference order per input kind and mode (first available wins).
PREFERENCES: dict[str, dict[str, list[str]]] = {
    "photo": {"auto": ["sharp", "moge_rgbd", "depth_anything_v2"],
              "quality": ["sharp", "moge_rgbd", "depth_anything_v2"],
              "fast": ["moge_rgbd", "depth_anything_v2", "sharp"]},
    "video:static_scene": {"auto": ["sharp", "moge_rgbd", "depth_anything_v2"],
                           "quality": ["sharp", "moge_rgbd", "depth_anything_v2"],
                           "fast": ["moge_rgbd", "depth_anything_v2", "sharp"]},
    "video:static_camera_dynamic": {m: ["moge_rgbd", "depth_anything_v2"] for m in MODES},
    "video:moving_camera": {"auto": ["recon3d_video"], "quality": ["recon3d_video"], "fast": ["recon3d_video"]},
}

# If nothing native can run: degrade honestly to a single-frame reconstruction.
FALLBACK = {"video:moving_camera": "video:static_scene"}


@dataclass
class Selection:
    backend: Backend | None
    input_kind: str
    effective_kind: str
    mode: str
    rejected: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"backend": self.backend.id if self.backend else None, "input_kind": self.input_kind,
                "effective_kind": self.effective_kind, "mode": self.mode, "rejected": self.rejected,
                "notes": self.notes}


def input_kind(inp: JobInput) -> str:
    return "photo" if inp.kind == "photo" else f"video:{inp.video_kind}"


def select(inp: JobInput, hw: HardwareReport | None, ctx: BackendContext, mode: str = "auto",
           forced: str | None = None) -> Selection:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    kind = input_kind(inp)
    backends = {b.id: b for b in all_backends()}
    options = {"mode": mode}
    sel = Selection(None, kind, kind, mode)
    if forced:
        b = backends.get(forced)
        if b is None:
            sel.notes.append(f"unknown backend '{forced}'")
            return sel
        ok, reasons = b.can_run(inp, hw, ctx, options)
        if ok:
            sel.backend = b
        else:
            sel.rejected.append({"backend": b.id, "reasons": reasons})
        return sel
    for b in backends.values():  # evaluated research backends: always explain why not
        if b.maturity == "unsupported" and kind in b.inputs:
            sel.rejected.append({"backend": b.id, "reasons": [b.description]})
    kinds = [kind] + ([FALLBACK[kind]] if kind in FALLBACK else [])
    for k in kinds:
        for bid in PREFERENCES.get(k, {}).get(mode, []):
            b = backends[bid]
            probe = JobInput(inp.path, inp.kind, inp.work_dir, inp.frames,
                             k.split(":", 1)[1] if ":" in k else None, inp.fps, inp.analysis)
            ok, reasons = b.can_run(probe, hw, ctx, options)
            if ok:
                sel.backend = b
                sel.effective_kind = k
                if k != kind:
                    sel.notes.append(f"No multi-view backend available for {kind}; falling back to a "
                                     f"single-frame reconstruction of the sharpest frame ({b.display_name}).")
                return sel
            sel.rejected.append({"backend": bid, "reasons": reasons})
    return sel
