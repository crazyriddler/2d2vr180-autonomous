"""User-facing component catalogue: what can be installed, what it enables,
its size and licence, and one call to install it (runtime or model)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .backends.base import BackendContext


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    kind: str                 # "runtime" | "model"
    target: str               # runtime id or model id
    feature: str
    recommended: bool
    requires: tuple[str, ...] = ()
    note: str = ""


COMPONENTS: list[Component] = [
    Component("engine-photo", "Photo & fixed-camera video engine", "runtime", "photo-cu128",
              "Required for every photo backend, fixed-camera video and the GPU VR180 renderer "
              "(PyTorch + CUDA, installed privately for 2D2VR180).", True),
    Component("model-moge-l", "MoGe-2 Large — quality metric depth", "model", "moge-2-vitl-normal",
              "Quality mode for photos and fixed-camera video (metric scale, textured mesh).", True,
              ("engine-photo",)),
    Component("model-moge-s", "MoGe-2 Small — fast metric depth", "model", "moge-2-vits-normal",
              "Fast mode for photos and fixed-camera video.", True, ("engine-photo",)),
    Component("model-da2", "Depth-Anything-V2 Small — commercial-safe depth", "model", "depth-anything-v2-small",
              "Apache-2.0 fallback usable in the Commercial licence profile (relative depth).", True,
              ("engine-photo",)),
    Component("model-sharp", "Apple SHARP — best single-photo 3D (research only)", "model", "sharp",
              "Feed-forward 3D Gaussians from one photo. Apple research licence: non-commercial.", False,
              ("engine-photo",)),
    Component("engine-video", "Moving-camera video engine (recon3d + VGGT + gsplat)", "runtime", "recon3d-cu124",
              "Multi-view reconstruction of videos where the camera moves. Needs an NVIDIA GPU with 12 GB+.", False,
              note="VGGT-1B (non-commercial) and MoGe-2 weights are downloaded automatically on first use."),
]


def get_component(cid: str) -> Component:
    for c in COMPONENTS:
        if c.id == cid:
            return c
    raise KeyError(cid)


def component_status(ctx: BackendContext, c: Component) -> dict:
    if c.kind == "runtime":
        st = ctx.runtimes.status(c.target)
        return {"installed": st["installed"], "size_gb": st.get("approx_size_gb"), "license": "various (see notices)",
                "license_url": "", "commercial_use": None, "detail": st.get("manifest_status", "")}
    st = ctx.models.status(c.target)
    size = st["size_bytes"] / 2**30 if st["size_bytes"] else None
    return {"installed": st["installed"], "size_gb": size, "license": st["license"], "license_url": st["license_url"],
            "commercial_use": st["commercial_use"], "detail": st.get("notes", "")}


def install_order(selected: list[str]) -> list[Component]:
    """Dependencies first, each component once."""
    out: list[Component] = []

    def add(cid: str) -> None:
        c = get_component(cid)
        for d in c.requires:
            add(d)
        if c not in out:
            out.append(c)

    for cid in selected:
        add(cid)
    return out


def install_component(ctx: BackendContext, c: Component, progress: Callable[[float | None, str], None],
                      cancel: Callable[[], bool], hf_token: str | None = None) -> None:
    """Install one component. Model licences must already be accepted by the caller (UI)."""
    if c.kind == "runtime":
        ctx.runtimes.install(c.target, lambda line: progress(None, line), cancel)
        return

    def prog(done: int, total: int | None) -> None:
        progress(done / total if total else None, f"{done / 2**20:.0f} MiB" + (f" / {total / 2**20:.0f} MiB" if total else ""))

    ctx.models.download(c.target, progress=prog, cancel=cancel, hf_token=hf_token or None)


def remove_component(ctx: BackendContext, c: Component) -> None:
    if c.kind == "runtime":
        ctx.runtimes.remove(c.target)
    else:
        ctx.models.delete(c.target)
