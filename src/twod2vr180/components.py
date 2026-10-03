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
    # ---------------------------------------------------------------- photos
    Component("engine-photo", "Photo engine", "runtime", "photo-cu128",
              "Required for every single-photo backend, fixed-camera video and the GPU VR180 renderer "
              "(PyTorch + CUDA, installed privately for 2D2VR180).", True),
    Component("model-moge-l", "MoGe-2 Large — quality metric depth", "model", "moge-2-vitl-normal",
              "Quality mode for photos and fixed-camera video (metric scale, textured mesh); also gives the "
              "multi-view engine its real-world scale.", True, ("engine-photo",)),
    Component("model-moge-s", "MoGe-2 Small — fast metric depth", "model", "moge-2-vits-normal",
              "Fast mode for photos and fixed-camera video.", True, ("engine-photo",)),
    Component("model-da2", "Depth-Anything-V2 Small — commercial-safe depth", "model", "depth-anything-v2-small",
              "Apache-2.0 fallback usable in the Commercial licence profile (relative depth).", True,
              ("engine-photo",)),
    Component("model-sharp", "Apple SHARP — best direct single-photo 3D (research only)", "model", "sharp",
              "Feed-forward 3D Gaussians from one photo. Apple research licence: non-commercial.", True,
              ("engine-photo",)),
    # ---------------------------------------------------------------- multi-view
    Component("engine-video", "Multi-view engine (several photos, moving-camera video)", "runtime", "recon3d-cu124",
              "Turns several photos of one scene, or a video where the camera moves, into one trained 3D "
              "Gaussian splat (VGGT + gsplat). Also used by Generative 3D. Needs an NVIDIA GPU with 10 GB+.", True),
    Component("model-vggt", "VGGT-1B — camera poses and depth from many views (non-commercial)", "model", "vggt-1b",
              "Finds where each photo/frame was taken and its depth.", True, ("engine-video",)),
    Component("model-da3", "Depth Anything 3 (Nested Giant-Large) — better multi-view poses and depth "
              "(non-commercial)", "model", "da3-nested-giant-large",
              "Finds where each view was taken and its depth more accurately than VGGT, with real-world scale. "
              "Used automatically by the multi-view engine when installed. ~6.8 GB.", True, ("engine-video",)),
    # ---------------------------------------------------------------- generative
    Component("engine-gen", "Generative engine (Qwen-Image-Edit, Wan 2.2, Stable Virtual Camera, LaMa)", "runtime", "gen-cu128",
              "Invents the parts of a scene a photo does not show (other sides of a subject, the surroundings "
              "for VR180) and fills holes behind objects with AI.", True),
    Component("model-lama", "LaMa — AI hole filling for VR180", "model", "big-lama",
              "Fills the gaps that appear behind objects in VR180 stereo with plausible texture.", True,
              ("engine-gen",)),
    Component("model-wan", "Wan 2.2 Fun 5B Control-Camera — generative 3D (Apache-2.0)", "model",
              "wan2.2-fun-5b-camera", "Video model that films new camera moves around your photo for Generative 3D. "
              "Keeps people and animals intact. ~25 GB; needs ~32 GB of RAM.", True,
              ("engine-gen", "engine-video", "model-vggt", "model-moge-l")),
    Component("model-qwen", "Qwen-Image-Edit-2511 + Multiple-Angles — sharp generative views (Apache-2.0)", "model",
              "qwen-image-edit-2511-q5", "Image model that redraws your photo from other camera angles (45°, 90°, "
              "the back, above, below) at about 1 megapixel — sharper than video frames. ~33 GB with its text "
              "encoder and LoRAs; needs ~32 GB of RAM. Needed for 'Real 3D from one photo'.", True,
              ("engine-gen", "model-qwen-base", "model-qwen-angles", "model-qwen-lightning",
               "model-qwen-lightning8", "engine-video", "model-vggt", "model-moge-l")),
    Component("model-qwen-base", "Qwen-Image-Edit-2511 text encoder and VAE (for Qwen views)", "model",
              "qwen-image-edit-2511-base", "Qwen2.5-VL-7B text/image encoder, VAE and configuration.", False,
              ("engine-gen",)),
    Component("model-qwen-angles", "Multiple-Angles LoRA (for Qwen views)", "model", "qwen-edit-2511-angles-lora",
              "Camera-angle control: 8 directions × 4 heights × 3 distances.", False, ("engine-gen",)),
    Component("model-qwen-lightning", "Qwen-Image-Edit Lightning LoRA (4 steps)", "model", "qwen-edit-2511-lightning",
              "Makes each Qwen view take 4 sampling steps instead of 40.", False, ("engine-gen",)),
    Component("model-qwen-lightning8", "Qwen-Image-Edit Lightning LoRA (8 steps, Quality)", "model",
              "qwen-edit-2511-lightning-8", "Used in Quality mode: 8 sampling steps per view for finer detail.",
              False, ("engine-gen",)),
    Component("model-seva", "Stable Virtual Camera 1.1 — alternative generative engine (non-commercial, gated)",
              "model", "seva-1.1", "Multi-view diffusion for scenes and objects without people (used only when Wan "
              "2.2 is not installed). Distorts people.", False,
              ("engine-gen", "model-seva-vae", "model-clip-h", "engine-video", "model-vggt", "model-moge-l"),
              note="Gated: first accept the licence at huggingface.co/stabilityai/stable-virtual-camera and paste "
                   "a Hugging Face access token (read) in Settings."),
    Component("model-seva-vae", "Stable Diffusion 2.1 VAE (for Stable Virtual Camera)", "model", "sd21-vae",
              "Image encoder/decoder used by Stable Virtual Camera.", False, ("engine-gen",)),
    Component("model-clip-h", "OpenCLIP ViT-H/14 (for Stable Virtual Camera)", "model", "clip-vit-h-14",
              "Image understanding used by Stable Virtual Camera.", False, ("engine-gen",)),
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
