"""FlashWorld camera conventions (workers/flashworld_worker.py), checked against FlashWorld's own
camera normalisation and ray code (workers/flashworld/fw_utils.py). Needs torch."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("plyfile")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import flashworld_worker as fw  # noqa: E402
from flashworld.fw_utils import create_rays, normalize_cameras  # noqa: E402


def test_swing_path_starts_at_the_photo_and_looks_at_the_subject():
    c2ws = fw.swing_path(2.0, n=24, swing_deg=30, rise_deg=6)
    assert len(c2ws) == 24 and np.allclose(c2ws[0], np.eye(4))
    for m in c2ws:
        to_pivot = np.array([0, 0, 2.0]) - m[:3, 3]
        assert np.dot(m[:3, 2], to_pivot / np.linalg.norm(to_pivot)) > 0.999      # looks at the subject
        assert abs(np.linalg.norm(to_pivot) - 2.0) < 1e-6                         # on the orbit
    yaw = [math.degrees(math.atan2(m[0, 2], m[2, 2])) for m in c2ws]
    assert max(yaw) == pytest.approx(30, abs=0.5) and min(yaw) == pytest.approx(-30, abs=0.5)
    assert np.allclose(c2ws[12][:3, 3], 0, atol=1e-6)                             # back at the photo halfway


def test_flashworld_rays_land_where_our_cameras_see_them():
    """A pixel at a known depth in our (OpenCV) camera k must come out of FlashWorld's own pipeline
    (normalised cameras, OpenGL rays) and to_photo_frame() at the same 3-D point."""
    W, H, fx, fy, cx, cy = 48, 64, 60.0, 60.0, 24.0, 32.0
    c2ws = fw.swing_path(2.5, n=8, swing_deg=35, rise_deg=8)
    cams = torch.from_numpy(fw.flashworld_cameras(c2ws, fx, fy, cx, cy, W, H))[None]
    norm, _, t_norm = normalize_cameras(cams, return_meta=True)
    rays_o, rays_d = create_rays(norm[0], H, W)                    # (V, H, W, 3) in FlashWorld's frame
    rng = np.random.default_rng(0)
    for k in (0, 2, 5):
        for _ in range(5):
            u, v = int(rng.integers(0, W)), int(rng.integers(0, H))
            depth = float(rng.uniform(1.0, 4.0))                    # z in our camera k
            d_cam = np.array([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, 1.0])
            ours = c2ws[k][:3, :3] @ (d_cam * depth) + c2ws[k][:3, 3]
            # FlashWorld's ray through the same pixel, at the same distance (in its normalised units)
            dist = np.linalg.norm(d_cam * depth) / (float(t_norm) + 1e-2)
            p = (rays_o[k, v, u] + rays_d[k, v, u] * dist).numpy()
            got, _, _ = fw.to_photo_frame(p[None], np.ones((1, 3)), np.array([[1.0, 0, 0, 0]]), c2ws[0],
                                          float(t_norm))
            assert np.allclose(got[0], ours, atol=1e-4), (k, u, v)


def test_crop_keeps_the_principal_point_mapping():
    img = np.zeros((300, 200, 3), np.uint8)
    out, s, x0, y0 = fw.crop_to(img, 48, 64)
    assert out.shape == (64, 48, 3)
    assert (200 / 2 - x0) * s == pytest.approx(24, abs=0.5) and (300 / 2 - y0) * s == pytest.approx(32, abs=0.5)
