"""Backend registry."""

from __future__ import annotations

from .base import Backend
from .photo import DepthAnythingBackend, MoGeRGBDBackend, SharpBackend
from .multiview import MultiViewBackend
from .video import RESEARCH_BACKENDS


def all_backends() -> list[Backend]:
    return [SharpBackend(), MoGeRGBDBackend(), DepthAnythingBackend(), MultiViewBackend(), *RESEARCH_BACKENDS]


def get_backend(backend_id: str) -> Backend:
    for b in all_backends():
        if b.id == backend_id:
            return b
    raise KeyError(f"unknown backend '{backend_id}'")
