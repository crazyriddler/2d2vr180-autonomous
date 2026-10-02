"""Geometry helpers of workers/multiview_worker.py (no GPU needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import multiview_worker as mv  # noqa: E402


def _rot(ax, ang):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def test_umeyama_recovers_sim3_with_outliers():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(500, 3))
    R = _rot([1, 2, 3], 0.7)
    dst = 2.5 * src @ R.T + [1, -2, 3]
    dst[:40] += rng.normal(0, 5, (40, 3))          # outliers
    s, R2, t = mv.robust_sim3(src, dst)
    assert abs(s - 2.5) < 1e-3 and np.allclose(R2, R, atol=1e-3) and np.allclose(t, [1, -2, 3], atol=1e-2)


def test_vggt_preprocessing_maps_intrinsics_back_to_the_original_image():
    for (h0, w0) in [(480, 640), (1080, 1920), (1600, 900)]:   # landscape, wide, portrait (cropped)
        img = np.zeros((h0, w0, 3), np.uint8)
        arr, info = mv.vggt_input(img)
        assert arr.shape[1] == 518 and arr.shape[0] % 14 == 0 and arr.shape[0] <= 518
        # a pinhole camera of the original image, expressed in VGGT pixels, must map back exactly
        f0, cx0, cy0 = 0.8 * w0, w0 / 2 + 7, h0 / 2 - 5
        new_h = round(h0 * 518 / w0 / 14) * 14
        Kv = np.array([[f0 / info["sx"], 0, cx0 / info["sx"]],
                       [0, f0 / info["sy"], cy0 / info["sy"] - info["crop"]], [0, 0, 1]])
        pad = 3
        Kv_p = Kv.copy()
        Kv_p[1, 2] += pad
        K = mv.to_original_K(Kv_p, info, pad)
        assert np.allclose(K[0], [f0, 0, cx0]) and np.allclose(K[1], [0, f0, cy0])
        assert abs(info["sy"] - h0 / new_h) < 1e-12


def test_unproject_round_trip():
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, :3] = _rot([0, 1, 0], 0.3)
    c2w[:3, 3] = [0.5, 0, -1]
    depth = np.full((48, 64), 2.0)
    pts = mv.unproject(depth, K, c2w)
    pc = (pts - c2w[:3, 3]) @ c2w[:3, :3]
    assert np.allclose(pc[..., 2], 2.0)
    assert np.allclose(pc[10, 20, 0] / 2 * 100 + 32, 20.5)
