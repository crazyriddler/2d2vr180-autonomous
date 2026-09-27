import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
FAKE_WORKER = REPO / "tests" / "fakes" / "fake_moge_worker.py"


@pytest.fixture(autouse=True)
def app_home(tmp_path, monkeypatch):
    home = tmp_path / "apphome"
    monkeypatch.setenv("TWOD2VR180_HOME", str(home))
    return home


@pytest.fixture
def rtx4080():
    from twod2vr180.hardware import GpuInfo, HardwareReport

    return HardwareReport(os="Windows", os_version="10.0.26100", python="3.11", cpu="test", cpu_count=16,
                          ram_total_mib=65536, disk_free_mib=500_000, disk_path="C:/", ffmpeg="ffmpeg",
                          gpus=[GpuInfo(0, "NVIDIA GeForce RTX 4080", 16376, 15000, "576.02", "8.9", "12.9")])


@pytest.fixture
def no_gpu(rtx4080):
    rtx4080.gpus = []
    return rtx4080


class FakeRuntimes:
    """Every runtime 'installed' and backed by the test interpreter."""

    def __init__(self):
        self.specs = {}

    def is_installed(self, rid):
        return True

    def python(self, rid):
        return Path(sys.executable)

    def worker_env(self, rid):
        return dict(os.environ)


def install_fake_model(mm, model_id, content=b"fake-weights"):
    e = mm.entries[model_id]
    d = mm.model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in e.files:
        (d / f.path).write_bytes(content)
        import hashlib

        files[f.path] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content), "hash_status": "tofu"}
    (d / "state.json").write_text(json.dumps({"files": files, "revision": "test",
                                              "license_accepted": {"license": e.license}}))


@pytest.fixture
def ctx(app_home, monkeypatch):
    from twod2vr180.backends.base import BackendContext
    from twod2vr180.backends.photo import MoGeRGBDBackend
    from twod2vr180.models import ModelManager

    monkeypatch.setattr(MoGeRGBDBackend, "worker_script", str(FAKE_WORKER))
    mm = ModelManager()
    for mid in ("moge-2-vits-normal", "moge-2-vitl-normal"):
        install_fake_model(mm, mid)
    return BackendContext(mm, FakeRuntimes(), "personal_research")


def synthetic_image(w=160, h=120, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.stack([xs * 255 // w, ys * 255 // h, np.full_like(xs, 90)], -1).astype(np.uint8)
    img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = [250, 250, 250]
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def photo(tmp_path):
    from PIL import Image

    p = tmp_path / "photo.jpg"
    Image.fromarray(synthetic_image()).save(p, quality=95)
    return p


def write_video(path, frames, fps=15):
    from twod2vr180.media import find_ffmpeg

    ff = find_ffmpeg()
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "12", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def textured_world(w=640, h=240, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8, 3)).astype(np.uint8)
    return np.kron(base, np.ones((8, 8, 1), np.uint8))


@pytest.fixture
def videos(tmp_path):
    world = textured_world()
    n = 45
    static = [world[:, 100:260].copy() for _ in range(n)]
    dynamic = []
    for i in range(n):
        f = world[:, 100:260].copy()
        x = 10 + i * 2
        f[40:80, x:x + 30] = [255, 0, 0]
        dynamic.append(f)
    pan = [world[:, 40 + i * 2: 200 + i * 2].copy() for i in range(n)]
    out = {}
    for name, frames in (("static", static), ("dynamic", dynamic), ("pan", pan)):
        out[name] = write_video(tmp_path / f"{name}.mp4", frames)
    return out
