"""Splat videos (.splv) in the app: the codec shared with the workers (workers/splv_codec.py, numpy + zlib
only) loaded once, plus conversions to the app's scene model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .paths import workers_dir
from .scene import INFERRED, Camera, GaussianScene

_codec = None


def codec():
    """The splv_codec module from the workers folder (the app and the workers share one implementation)."""
    global _codec
    if _codec is None:
        spec = importlib.util.spec_from_file_location("twod2vr_splv_codec", workers_dir() / "splv_codec.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _codec = mod
    return _codec


def is_splv(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"SPLV"
    except OSError:
        return False


def camera_of(header: dict) -> Camera:
    c = header["camera"]
    return Camera(int(c["width"]), int(c["height"]), float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]))


def frame_scene(reader, k: int) -> GaussianScene:
    """Frame ``k`` of a splat video as a scene (camera frame of the filming camera, metres)."""
    f = reader.frame(k)
    n = len(f["opacity"])
    return GaussianScene(f["means"], f["scales"], f["quats"], f["opacity"], f["colors"],
                         np.full(n, INFERRED, np.uint8), cameras=[camera_of(reader.header)], metric_scale=True)


def open_reader(path: Path):
    return codec().Reader(path)
