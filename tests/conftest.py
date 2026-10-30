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
    monkeypatch.setenv("TWOD2VR_HOME", str(home))
    return home


@pytest.fixture
def rtx4080():
    from twod2vr.hardware import GpuInfo, HardwareReport

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

    def status(self, rid):
        return {"id": rid, "installed": True, "approx_size_gb": 1.0, "manifest_status": "fake", "description": rid}

    def install(self, rid, log=print, cancel=None):
        log(f"fake install {rid}")

    def remove(self, rid):
        pass

    def unused(self):
        return []

    def delete_unused(self):
        pass


def install_fake_model(mm, model_id, content=b"fake-weights"):
    e = mm.entries[model_id]
    d = mm.model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in e.files:
        (d / f.path).parent.mkdir(parents=True, exist_ok=True)
        (d / f.path).write_bytes(content)
        import hashlib

        files[f.path] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content), "hash_status": "tofu"}
    (d / "state.json").write_text(json.dumps({"files": files, "revision": "test",
                                              "license_accepted": {"license": e.license}}))


FAKES = REPO / "tests" / "fakes"


@pytest.fixture
def ctx(app_home, monkeypatch):
    """Real 3D's engine (fake workers in the test interpreter) with SHARP and MoGe-3 'installed'."""
    from twod2vr.backends.base import BackendContext
    from twod2vr.backends.locked import LockedPhotoBackend
    from twod2vr.backends.video_splat import SplatVideoBackend
    from twod2vr.models import ModelManager

    monkeypatch.setattr(LockedPhotoBackend, "geometry_script", str(FAKE_WORKER))
    monkeypatch.setattr(LockedPhotoBackend, "worker_script", str(FAKES / "fake_sharp_depth_worker.py"))
    monkeypatch.setattr(LockedPhotoBackend, "render_script", str(FAKES / "fake_render_worker.py"))
    monkeypatch.setattr(LockedPhotoBackend, "refine_script", str(FAKES / "fake_refine_worker.py"))
    monkeypatch.setattr(SplatVideoBackend, "worker_script", str(FAKES / "fake_video_splat_worker.py"))
    mm = ModelManager()
    for mid in ("sharp", "moge-3-vitl"):
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
    from twod2vr.media import find_ffmpeg

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


# ---------------------------------------------------------------- splat videos (.splv) for player/viewer tests
FRAMES, FPS = 12, 6.0


def _plane(z, color, n, half, offset=(0.0, 0.0)):
    from twod2vr.scene import Camera, GaussianScene

    xs, ys = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    m = np.stack([xs.ravel() + offset[0], ys.ravel() + offset[1], np.full(n * n, z)], 1).astype(np.float32)
    k = len(m)
    return GaussianScene(m, np.full((k, 3), 2 * half / n, np.float32), np.tile(np.float32([1, 0, 0, 0]), (k, 1)),
                         np.ones(k, np.float32), np.tile(np.float32(color), (k, 1)),
                         cameras=[Camera(160, 120, 150, 150, 80, 60)])


def make_video(path, audio=None):
    """A splat video: a blue wall 3 m away and a red square 1.5 m away moving 2 m to the right."""
    from twod2vr.splv import codec

    c = codec()
    quant = c.default_quant(90, 75)
    bg = _plane(3.0, [0.2, 0.4, 0.9], 60, 2.0)

    def chunks():
        prev = None
        for k in range(FRAMES):
            sq = _plane(1.5, [1, 0.1, 0.1], 20, 0.3, (-1 + 2.0 * k / (FRAMES - 1), 0))
            arr = c.quantize(np.concatenate([bg.means, sq.means]), np.concatenate([bg.scales, sq.scales]),
                             np.concatenate([bg.quats, sq.quats]), np.concatenate([bg.colors, sq.colors]),
                             np.concatenate([bg.opacity, sq.opacity]), quant)
            key = k % 4 == 0
            yield c.encode_frame(arr, None if key else prev), key
            prev = arr

    header = {"fps": FPS, "count": 60 * 60 + 20 * 20, "keyframe_interval": 4, "quant": quant,
              "camera": {"width": 160, "height": 120, "fx": 150, "fy": 150, "cx": 80, "cy": 60}}
    c.write_file(path, header, chunks(), audio=audio[0] if audio else None,
                 audio_mime=audio[1] if audio else "audio/mp4")
    return path


