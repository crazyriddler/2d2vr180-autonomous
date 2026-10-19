"""Fitting a photo's splat to the photo (workers/refine.py), with a tiny differentiable renderer in place of
gsplat (CUDA only). Needs torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
plyfile = pytest.importorskip("plyfile")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

SH_C0 = 0.28209479177387814


def tiny_render(means, quats, scales, opacities, sh, viewmat, K, width, height, sh_degree):
    """Isotropic splats, sorted front to back, alpha-composited (enough to test the fitting)."""
    order = torch.argsort(means[:, 2])
    m, s, o = means[order], scales[order], opacities[order]
    col = (0.5 + SH_C0 * sh[order, 0]).clamp(min=0)
    u = K[0, 0] * m[:, 0] / m[:, 2] + K[0, 2]
    v = K[1, 1] * m[:, 1] / m[:, 2] + K[1, 2]
    sig = K[0, 0] * s.mean(1) / m[:, 2]
    yy, xx = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    d2 = (xx[None] - u[:, None, None]) ** 2 + (yy[None] - v[:, None, None]) ** 2
    a = (o[:, None, None] * torch.exp(-0.5 * d2 / sig[:, None, None] ** 2)).clamp(max=0.99)
    T = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1 - a[:-1]]), 0)
    wgt = T * a
    rgb = (wgt[..., None] * col[:, None, None, :]).sum(0)
    return rgb, wgt.sum(0)


def write_ply(path, means, colors, scale, opacity):
    n = len(means)
    dt = [(k, "f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
                               "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    v = np.zeros(n, dt)
    v["x"], v["y"], v["z"] = means.T
    dc = (colors - 0.5) / SH_C0
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc.T
    v["opacity"] = np.log(opacity / (1 - opacity))
    for i in range(3):
        v[f"scale_{i}"] = np.log(scale)
    v["rot_0"] = 1
    extra = np.array([(1.0,)], [("frame", "f4")])
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex"), plyfile.PlyElement.describe(extra, "meta")]).write(path)


def test_the_fit_brings_the_render_to_the_photo_without_moving_anything(tmp_path):
    import refine

    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(-0.45, 0.45, 8), np.linspace(-0.45, 0.45, 8))
    means = np.stack([gx.ravel(), gy.ravel(), np.full(64, 2.0)], 1)
    truth = rng.uniform(0.1, 0.9, (64, 3))
    f, W, H = 64.0, 48, 48
    # the photo: the true colours
    with torch.no_grad():
        sh = torch.tensor((truth - 0.5) / SH_C0, dtype=torch.float32)[:, None]
        Kt = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32)
        photo, _ = tiny_render(torch.tensor(means, dtype=torch.float32), None, torch.full((64, 3), 0.06),
                               torch.full((64,), 0.9), sh, None, Kt, W, H, 0)
    photo = (photo.clamp(0, 1).numpy() * 255).astype(np.uint8)
    # the splat as predicted: colours off, a little too small and too transparent
    write_ply(tmp_path / "in.ply", means, np.clip(truth + rng.normal(0, 0.15, truth.shape), 0.05, 0.95), 0.05, 0.7)
    st = refine.refine_ply(tmp_path / "in.ply", tmp_path / "out.ply", photo, f, (W, H), torch, torch.device("cpu"),
                           iters=150, render=tiny_render)
    assert st["used"] and st["psnr_after"] > st["psnr_before"] + 3
    out = plyfile.PlyData.read(tmp_path / "out.ply")
    v = out["vertex"].data
    assert np.allclose(np.stack([v["x"], v["y"], v["z"]], 1), means, atol=1e-6)      # nothing moved
    assert "meta" in out                                                              # metadata kept
    assert np.abs(v["scale_0"] - np.log(0.05)).max() <= 0.3 + 1e-5                   # sizes bounded


def test_no_improvement_keeps_the_original(tmp_path):
    import refine

    means = np.array([[0.0, 0.0, 2.0]])
    write_ply(tmp_path / "in.ply", means, np.array([[0.5, 0.5, 0.5]]), 0.05, 0.9)
    with torch.no_grad():
        Kt = torch.tensor([[64.0, 0, 16], [0, 64.0, 16], [0, 0, 1]])
        photo, _ = tiny_render(torch.tensor(means, dtype=torch.float32), None, torch.full((1, 3), 0.05),
                               torch.full((1,), 0.9), torch.zeros((1, 1, 3)), None, Kt, 32, 32, 0)
    photo = (photo.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    st = refine.refine_ply(tmp_path / "in.ply", tmp_path / "out.ply", photo, 64.0, (32, 32), torch,
                           torch.device("cpu"), iters=20, render=tiny_render)
    assert not st["used"] and not (tmp_path / "out.ply").exists()
