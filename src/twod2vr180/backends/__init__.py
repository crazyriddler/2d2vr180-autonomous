"""Backend registry."""

from __future__ import annotations

from .base import Backend
from .generative import GenerativeSceneBackend
from .locked import LockedPhotoBackend
from .photo import DepthAnythingBackend, MoGeRGBDBackend, SharpBackend
from .multiview import MultiViewBackend
from .video import RESEARCH_BACKENDS
from .video4d import Video4DBackend


def all_backends() -> list[Backend]:
    return [SharpBackend(), MoGeRGBDBackend(), DepthAnythingBackend(), MultiViewBackend(), GenerativeSceneBackend(),
            LockedPhotoBackend(),
            Video4DBackend(), *RESEARCH_BACKENDS]


def get_backend(backend_id: str) -> Backend:
    for b in all_backends():
        if b.id == backend_id:
            return b
    raise KeyError(f"unknown backend '{backend_id}'")
