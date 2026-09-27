"""Backend registry."""

from __future__ import annotations

from .base import Backend
from .photo import DepthAnythingBackend, MoGeRGBDBackend, SharpBackend
from .video import RESEARCH_BACKENDS, Recon3DBackend


def all_backends() -> list[Backend]:
    return [SharpBackend(), MoGeRGBDBackend(), DepthAnythingBackend(), Recon3DBackend(), *RESEARCH_BACKENDS]


def get_backend(backend_id: str) -> Backend:
    for b in all_backends():
        if b.id == backend_id:
            return b
    raise KeyError(f"unknown backend '{backend_id}'")
