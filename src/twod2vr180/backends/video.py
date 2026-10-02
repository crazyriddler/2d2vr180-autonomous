"""Registry of evaluated-but-not-shippable research backends (the multi-view
backend lives in multiview.py)."""

from __future__ import annotations

from .base import MULTI_VIEW, NOVEL_VIEW_COMPLETION, SINGLE_VIEW, Backend


class ResearchOnlyBackend(Backend):
    """Evaluated upstream projects that cannot run in this build. Registered so
    the UI and reports can say *why* instead of silently omitting them."""

    maturity = "unsupported"

    def __init__(self, id: str, name: str, upstream: str, caps: frozenset[str], inputs: tuple[str, ...],
                 reason: str):
        self.id = id
        self.display_name = name
        self.upstream = [upstream]
        self.capabilities = caps
        self.inputs = inputs
        self.description = reason

    def run(self, *a, **k):  # pragma: no cover - availability() prevents this
        raise NotImplementedError(self.description)


RESEARCH_BACKENDS = [
    ResearchOnlyBackend(
        "one2scene", "One2Scene", "one2scene", frozenset({SINGLE_VIEW, NOVEL_VIEW_COMPLETION}), ("photo",),
        "Unsupported on RTX 4080 16 GB for now: needs a ~19 GB denoise checkpoint plus SDXL-VAE, ZIM and "
        "GFPGAN stacks, repository has no LICENSE file (all rights reserved by default), Linux/CUDA only."),
    ResearchOnlyBackend(
        "longsplat", "LongSplat", "longsplat", frozenset({MULTI_VIEW}), ("video:moving_camera",),
        "Unsupported: NVIDIA Source Code License (non-commercial) + Inria 3DGS licence; requires compiling "
        "simple-knn / diff-gaussian-rasterization / fused-ssim CUDA extensions (MSVC + CUDA toolkit on Windows)."),
    ResearchOnlyBackend(
        "depthsplat", "DepthSplat", "depthsplat", frozenset({MULTI_VIEW}), ("video:moving_camera",),
        "Not integrated yet: MIT code (cvg/depthsplat), but requires posed input views and a dataset-style "
        "loader; evaluate after the recon3d path is GPU-validated."),
    ResearchOnlyBackend(
        "gsfixer", "GSFixer", "gsfixer", frozenset({NOVEL_VIEW_COMPLETION}), (),
        "Unsupported: repository has no LICENSE file, tested only on H20/H100 (CUDA 12.1), requires compiled "
        "rasteriser extensions and a video-diffusion model."),
    ResearchOnlyBackend(
        "gsfix3d", "GSFix3D", "gsfix3d", frozenset({NOVEL_VIEW_COMPLETION}), (),
        "Research-only: Gaussian-Splatting licence makes the project non-commercial; models under OpenRAIL++-M; "
        "requires per-scene fine-tuning of a diffusion model and compiled rasteriser extensions."),
]
