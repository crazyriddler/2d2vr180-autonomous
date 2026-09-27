import numpy as np
import pytest

from twod2vr180.scene import (INFERRED, Camera, GaussianScene, SceneError, load_scene, read_gaussian_ply,
                              write_gaussian_ply, write_splat)


def make_scene(n=500, seed=0, sh=True):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianScene(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        scales=rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32),
        quats=q, opacity=rng.uniform(0.1, 0.99, n).astype(np.float32),
        colors=rng.uniform(0, 1, (n, 3)).astype(np.float32),
        provenance=np.full(n, INFERRED, np.uint8),
        sh_rest=rng.normal(size=(n, 45)).astype(np.float32) if sh else None,
        cameras=[Camera(640, 480, 500, 500, 320, 240)], metric_scale=True)


def test_ply_roundtrip(tmp_path):
    s = make_scene()
    p = write_gaussian_ply(s, tmp_path / "s.ply")
    r = read_gaussian_ply(p)
    assert len(r) == len(s)
    np.testing.assert_allclose(r.means, s.means, atol=1e-6)
    np.testing.assert_allclose(r.scales, s.scales, rtol=1e-5)
    np.testing.assert_allclose(r.opacity, s.opacity, atol=1e-5)
    np.testing.assert_allclose(r.colors, s.colors, atol=1e-5)
    np.testing.assert_allclose(np.abs((r.quats * s.quats).sum(1)), 1, atol=1e-5)
    np.testing.assert_allclose(r.sh_rest, s.sh_rest)
    assert r.metric_scale and r.cameras[0].fx == 500
    assert r.validate() == []
    assert (p.with_suffix(".meta.json")).exists()


def test_ply_header_is_standard_3dgs(tmp_path):
    p = write_gaussian_ply(make_scene(sh=False), tmp_path / "s.ply")
    header = p.read_bytes().split(b"end_header")[0].decode()
    for prop in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"):
        assert f"property float {prop}\n" in header
    assert "provenance" not in header  # keep third-party viewers happy


def test_splat_roundtrip(tmp_path):
    s = make_scene(sh=False)
    p = write_splat(s, tmp_path / "s.splat")
    assert p.stat().st_size == 32 * len(s)
    r = load_scene(p)
    assert len(r) == len(s)
    # importance-sorted: compare as sets of positions
    a = np.sort(s.means[:, 0])
    b = np.sort(r.means[:, 0])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_sharp_style_extra_elements(tmp_path):
    """SHARP writes extra PLY elements after 'vertex' (intrinsic, image_size...)."""
    s = make_scene(n=10, sh=False)
    p = write_gaussian_ply(s, tmp_path / "base.ply")
    raw = p.read_bytes()
    head, body = raw.split(b"end_header\n")
    head += (b"element image_size 2\nproperty uint image_size\n"
             b"element intrinsic 9\nproperty float intrinsic\n")
    extra = np.array([800, 600], "<u4").tobytes() + np.array([700, 0, 400, 0, 700, 300, 0, 0, 1], "<f4").tobytes()
    q = tmp_path / "sharp.ply"
    q.write_bytes(head + b"end_header\n" + body + extra)
    q.with_suffix(".meta.json").unlink(missing_ok=True)
    r = read_gaussian_ply(q)
    assert len(r) == 10
    assert r.cameras and r.cameras[0].width == 800 and r.cameras[0].fx == pytest.approx(700)


def test_validate_detects_nan_and_explosion():
    s = make_scene(n=100, sh=False)
    s.means[3] = np.nan
    s.means[4] = 1e9
    probs = s.validate()
    assert any("NaN" in p for p in probs)
    clean = s.finite_subset()
    assert len(clean) == 99


def test_not_a_ply(tmp_path):
    p = tmp_path / "x.ply"
    p.write_text("hello")
    with pytest.raises(SceneError):
        read_gaussian_ply(p)
