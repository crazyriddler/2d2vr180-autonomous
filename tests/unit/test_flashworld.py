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


def test_flashworld_render_call_is_valid_for_gsplat_1_5_3(monkeypatch):
    """gsplat 1.5.3 (the version with Windows wheels) checks backgrounds against the projected means'
    leading dims, which are empty in packed mode: per-view backgrounds of shape (1, D) then fail with
    'AssertionError: torch.Size([1, 4])' (owner's rc34 run). FlashWorld must call it unpacked."""
    import types

    if "gsplat" not in sys.modules:
        try:
            import gsplat  # noqa: F401
        except ImportError:                                  # CPU test machines: the module only has to import
            monkeypatch.setitem(sys.modules, "gsplat", types.SimpleNamespace(rasterization=None))
    import flashworld.models.render as render

    def strict_rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                             render_mode="RGB", backgrounds=None, packed=True, **kw):
        C = viewmats.shape[0]
        channels = 3 + (1 if "D" in render_mode else 0)
        image_dims = () if packed else (C,)                 # what 1.5.3's rasterize_to_pixels derives
        if backgrounds is not None:
            bg = torch.cat([backgrounds, torch.zeros(C, 1)], -1) if "D" in render_mode else backgrounds
            assert bg.shape == image_dims + (channels,), bg.shape
        z = (means.sum() + quats.sum() + scales.sum() + opacities.sum() + colors.sum()) * 0
        return torch.zeros(C, height, width, channels) + z, torch.zeros(C, height, width, 1), {}

    monkeypatch.setattr(render, "rasterization", strict_rasterization)
    n, sh = 50, 2
    params = torch.cat([torch.randn(n, 3), torch.rand(n, 1), torch.rand(n, 3) * .1,
                        torch.nn.functional.normalize(torch.randn(n, 4), dim=-1), torch.randn(n, (sh + 1) ** 2 * 3)], -1)
    c2ws = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    intr = torch.tensor([[40.0, 40.0, 16.0, 12.0]]).repeat(1, 2, 1)
    rgb, depth = render.gaussian_render(params[None], c2ws, intr, 32, 24, sh_degree=sh, bg_mode="white")[:2]
    assert rgb.shape[-2:] == (24, 32)


def test_views_are_decoded_in_chunks_with_the_same_result():
    """GenerationSystem._chunk_frames: the 3D decoder and VAE encoder get k views at a time (16 GB cards)
    and the outputs are put back together in the original order and shape."""
    import types

    if "gsplat" not in sys.modules:
        try:
            import gsplat  # noqa: F401
        except ImportError:
            sys.modules["gsplat"] = types.SimpleNamespace(rasterization=None)
    from flashworld.system import GenerationSystem

    seen = []

    class Dec(torch.nn.Module):
        def forward(self, feats, z, cameras):           # (BT, C, 1, h, w), (BT, ...), (B, T, 11)
            seen.append(feats.shape[0])
            assert cameras.shape[1] == feats.shape[0]
            g = feats.mean((1, 2, 3, 4))[:, None, None] + z.mean((1, 2, 3, 4))[:, None, None] + cameras.flatten(0, 1)[:, None, :1]
            return g.expand(-1, 5, 7).unflatten(0, (cameras.shape[0], cameras.shape[1]))

    class Vae:
        def encode(self, x):
            seen.append(x.shape[0])
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=lambda: x * 2))

    T = 7
    feats, z, cams = torch.randn(T, 4, 1, 3, 3), torch.randn(T, 2, 1, 3, 3), torch.randn(1, T, 11)
    x = torch.randn(T, 3, 1, 8, 8)
    ref_dec, ref_enc = Dec()(feats, z, cams), Vae().encode(x).latent_dist.sample()
    sysm = types.SimpleNamespace(recon_decoder=Dec(), vae=Vae())
    seen.clear()
    GenerationSystem._chunk_frames(sysm, 3)
    assert torch.equal(sysm.recon_decoder(feats, z, cams), ref_dec)
    assert torch.equal(sysm.vae.encode(x).latent_dist.sample(), ref_enc)
    assert seen == [3, 3, 1, 3, 3, 1]
