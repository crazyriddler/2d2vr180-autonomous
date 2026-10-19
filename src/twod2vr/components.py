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
    # ---------------------------------------------------------------- Real 3D (photos and splat videos)
    Component("engine-photo", "Real 3D engine", "runtime", "photo-cu128",
              "Makes the 3D of photos and splat videos (PyTorch + CUDA, installed privately for 2D2VR).", True),
    Component("model-sharp", "Apple SHARP — the 3D Gaussians of every photo and frame (research only)", "model",
              "sharp", "Feed-forward 3D Gaussians from one picture, aligned to the measured depth. Apple research "
              "licence: non-commercial.", True, ("engine-photo",)),
    Component("model-moge3", "MoGe-3 Large — the measured depth, lens and real-world scale", "model", "moge-3-vitl",
              "Metric depth and field of view of every photo and video frame (MIT).", True,
              ("engine-photo", "model-sharp", "model-face", "model-sapiens-pointmap")),
    Component("model-moge3-giant", "MoGe-3 Giant — the most accurate depth for photos", "model", "moge-3-vitg",
              "MoGe-3 with a 4x bigger network (MIT, 5 GB): photos use it instead of the Large model when it is "
              "installed (splat videos keep the Large one, for speed).", True, ("engine-photo", "model-moge3")),
    Component("model-depth-pro", "Apple Depth Pro — a second measured depth for photos (research only)", "model",
              "depth-pro", "A second, independent metric depth with the sharpest object edges and its own lens "
              "estimate, joined to MoGe-3 (Apple research licence: non-commercial, 1.9 GB).", True,
              ("engine-photo",)),
    Component("model-face", "YuNet face finder — real-world scale from the eyes", "model", "yunet-face",
              "Finds the eyes of the people so the scene is scaled to the eye distance you set (MIT, 0.2 MB, runs "
              "on the CPU).", True, ("engine-photo",)),
    Component("model-sapiens-seg", "Sapiens2 body parts — where the people are", "model",
              "sapiens2-seg-0.4b", "Finds the people so their own 3D shape can be used (Meta Sapiens2, 1.6 GB).",
              True, ("engine-photo",)),
    Component("model-sapiens-pointmap", "Sapiens2 3D shape of people — real noses and faces", "model",
              "sapiens2-pointmap-1b", "People's 3D shape from a model trained only on humans (Meta Sapiens2, ICLR "
              "2026): faces, noses and bodies keep their real relief instead of coming out flat, and their distance "
              "helps set the real-world scale. 6.5 GB.", True, ("engine-photo", "model-sapiens-seg")),
    Component("model-sapiens-normal", "Sapiens2 surface normals of people — their fine relief", "model",
              "sapiens2-normal-1b", "The slopes of people's surfaces from a model trained only on humans (Meta "
              "Sapiens2): folds of clothes, fingers and the relief of a face, added to their 3D shape in photos. "
              "6.2 GB.", True, ("engine-photo", "model-sapiens-seg")),
    # ---------------------------------------------------------------- AI fill (photos)
    Component("engine-gen", "AI painter engine", "runtime", "gen-cu128",
              "Paints what a photo hides when you move your head (Real 3D fill).", True),
    Component("model-qwen", "Qwen-Image-Edit-2511 — paints what the photo hides (Apache-2.0)", "model",
              "qwen-image-edit-2511-q5", "Image model that fills the holes seen from moved positions, about 1 "
              "megapixel. ~30 GB with its text encoder; needs ~32 GB of RAM. Photos only.", True,
              ("engine-gen", "model-qwen-base", "model-qwen-lightning", "engine-photo", "model-sharp",
               "model-moge3")),
    Component("model-qwen-base", "Qwen-Image-Edit-2511 text encoder and VAE", "model",
              "qwen-image-edit-2511-base", "Qwen2.5-VL-7B text/image encoder, VAE and configuration.", True,
              ("engine-gen",)),
    Component("model-qwen-lightning", "Qwen-Image-Edit Lightning LoRA (4 steps)", "model", "qwen-edit-2511-lightning",
              "Makes each painting take 4 sampling steps instead of 40.", True, ("engine-gen",)),
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
