"""Backend registry: Real 3D for photos, splat videos for videos (2D2VR 2.0)."""

from __future__ import annotations

from .base import Backend
from .locked import LockedPhotoBackend
from .video_splat import SplatVideoBackend


def all_backends() -> list[Backend]:
    return [LockedPhotoBackend(), SplatVideoBackend()]


def get_backend(backend_id: str) -> Backend:
    for b in all_backends():
        if b.id == backend_id:
            return b
    raise KeyError(f"unknown backend '{backend_id}'")
