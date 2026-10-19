"""Precision for Real 3D photos (workers/precision.py). Needs torch (present in the GPU runtime)."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))


def _scene(h=160, w=200):
    """A wall 3 m away with 1 cm bumps every 12 px."""
    v, u = np.mgrid[0:h, 0:w].astype(float)
    return 3.0 + 0.1 * u / w + 0.01 * np.sin(u / 2.0) * np.sin(v / 2.0)


def test_normals_restore_the_relief_the_depth_lost():
    import precision as pr

    truth = _scene()
    f = 250.0
    smooth = pr.blur(truth, 6.0, torch)                    # what a depth model gives: no bumps
    normals = pr.normals_from_depth(truth, f)              # what a normal model gives: the bumps' slopes
    out, st = pr.integrate_normals(smooth, normals, f, torch, cutoff=20, iters=600)
    inner = (slice(8, -8), slice(8, -8))
    err_before = np.abs(smooth - truth)[inner].mean()
    err_after = np.abs(out - truth)[inner].mean()
    assert st["used"] and err_after < 0.35 * err_before
    bumps_truth = (truth - pr.blur(truth, 6.0, torch))[inner]
    bumps_out = (out - pr.blur(out, 6.0, torch))[inner]
    assert np.corrcoef(bumps_truth.ravel(), bumps_out.ravel())[0, 1] > 0.9
    # the overall shape and scale stay the depth's
    assert abs(np.median(out[inner]) - np.median(truth[inner])) < 0.005


def test_normals_from_another_convention_are_aligned_first():
    import precision as pr

    truth = _scene()
    n = pr.normals_from_depth(truth, 250.0)
    other = n * np.array([1.0, -1.0, -1.0])                # e.g. OpenGL axes (y up, z backwards)
    ref = pr.normals_from_depth(pr.blur(truth, 6.0, torch), 250.0)
    aligned, st = pr.align_normals(other, ref)
    assert st["ok"] and st["median_angle_deg"] < 25          # the bumps tilt by up to ~23°
    good = np.isfinite(n).all(-1)
    assert np.abs(np.abs((aligned[good] * n[good]).sum(-1)) - 1).max() < 2e-3          # within ~3°


def test_unrelated_normals_are_refused():
    import precision as pr

    truth = _scene()
    ref = pr.normals_from_depth(truth, 250.0)
    rng = np.random.default_rng(0)
    junk = rng.normal(size=ref.shape)
    junk /= np.linalg.norm(junk, axis=-1, keepdims=True)
    _, st = pr.align_normals(junk, ref)
    assert not st["ok"]


def test_relief_change_is_capped():
    import precision as pr

    truth = _scene()
    n = pr.normals_from_depth(truth + 0.4 * np.sin(np.arange(200) / 7.0)[None, :], 250.0)   # wrong normals
    out, st = pr.integrate_normals(truth, n, 250.0, torch, cutoff=20, limit=0.03)
    ok = np.isfinite(out)
    assert np.abs(np.log(out[ok]) - np.log(truth[ok])).max() <= 0.03 + 1e-9


def test_edges_go_to_one_side_by_colour():
    import precision as pr

    h, w = 60, 80
    truth = np.full((h, w), 4.0)
    truth[:, :40] = 1.5                                     # a person (left) in front of a wall
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :40] = (220, 120, 90)
    img[:, 40:] = (40, 60, 160)
    blurred = np.exp(pr.blur(np.log(truth), 2.0, torch))   # monocular depth: a ramp across the edge
    out, band, far, st = pr.resolve_edges(blurred, img, torch, rel=0.04, radius=3)
    assert st["band_share"] > 0
    cols = band.any(0)
    assert cols[37:43].all()
    # every pixel of the band is near one of the two surfaces, on the side of its colour
    assert np.abs(out[:, :40][band[:, :40]] - 1.5).max() < 0.15
    assert np.abs(out[:, 40:][band[:, 40:]] - 4.0).max() < 0.4
    assert np.isfinite(far[band]).all() and (far[band] > 2.5).all()
    # in-between depths are gone
    mid = (out > 2.0) & (out < 3.5)
    assert not mid.any()


def test_two_models_join_shape_and_edges():
    import precision as pr

    truth = np.full((80, 100), 4.0)
    truth[:, :50] = 2.0
    smooth = np.exp(pr.blur(np.log(truth), 4.0, torch))     # first model: right shape, soft edge
    sharp = truth * 1.2                                     # second model: sharp edge, other scale
    out, st = pr.depth_ensemble(smooth, sharp, sigma=8, torch=torch)
    assert st["used"] and abs(st["scale_second_to_primary"] - 1 / 1.2) < 0.01
    row = out[40]
    assert row[45] < 2.2 and row[55] > 3.6                 # the edge stays sharp
    assert abs(np.median(out[:, :40]) - 2.0) < 0.05 and abs(np.median(out[:, 60:]) - 4.0) < 0.1


def test_a_disagreeing_second_model_is_dropped():
    import precision as pr

    a = np.full((60, 60), 3.0)
    b = a.copy()
    b[:, :30] = 0.6                                         # half the picture 5x nearer: not the same scene
    out, st = pr.depth_ensemble(a, b, sigma=6, torch=torch, max_disagree=0.25)
    assert not st["used"] or np.allclose(out, a)


def test_passes_are_combined_by_median_after_scale():
    import precision as pr

    base = _scene(40, 50)
    passes = [base, base * 1.05, base * 0.97 + 0.3 * (np.arange(50)[None, :] == 10)]   # one with an outlier
    med, spread = pr.combine_passes(passes)
    assert np.abs(med - base)[:, 12:].max() < 0.02
    assert np.nanmax(spread) > 0


def test_the_lens_is_checked():
    import precision as pr

    fov, src, _ = pr.lens_choice(65.0, {"MoGe-3": 66.0, "Depth Pro": 64.5})
    assert fov == 65.0 and src == "EXIF"
    fov, src, note = pr.lens_choice(65.0, {"MoGe-3": 40.0, "Depth Pro": 41.0})     # a 1.6x crop/zoom
    assert src.startswith("models") and 40 <= fov <= 41 and "65.0" in note
    fov, src, _ = pr.lens_choice(65.0, {"MoGe-3": 40.0, "Depth Pro": 60.0})        # models disagree: EXIF
    assert fov == 65.0
    fov, src, _ = pr.lens_choice(None, {"MoGe-3": 50.0, "Depth Pro": 54.0})
    assert src == "models" and fov == pytest.approx(52.0)
    fov, src, _ = pr.lens_choice(None, {"MoGe-3": 50.0, "Depth Pro": None})
    assert fov == 50.0
