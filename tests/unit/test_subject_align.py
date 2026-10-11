"""Placing a generated full-subject mesh where the subject is in the photo (workers/subject_align.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import subject_align as sa  # noqa: E402


def box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
         [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v, np.array(f)


def figure():
    """An asymmetric 'person': body, head, one arm, a nose (front) - in TRELLIS-like units (z up)."""
    parts = [box((-0.15, -0.08, -0.45), (0.15, 0.08, 0.2)), box((-0.08, -0.08, 0.2), (0.08, 0.08, 0.38)),
             box((0.15, -0.05, -0.1), (0.3, 0.05, 0.15)), box((-0.02, -0.14, 0.27), (0.02, -0.08, 0.31))]
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def test_mesh_is_placed_where_the_subject_is_in_the_photo():
    v, f = figure()
    pts, _ = sa.sample_surface(v, f, 60000)
    # ground truth: the subject 2.4 m away, 1.7 m tall, turned 30° and seen from slightly above
    rots = sa.candidate_rotations(24)
    a = np.radians(10)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    R_gt = tilt @ rots[4 * 24 + 2]           # z up, turned 30°, plus a tilt no candidate has exactly
    s_gt, t_gt = 1.7 / 0.83, np.array([0.1, 0.05, 2.4])
    gt = {"scale": s_gt, "R": R_gt, "t": t_gt}
    W, H = 320, 240
    K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]])
    depth = sa.render_depth(sa.apply(pts, gt), K, np.eye(4), W, H, dilate=1)
    mask = depth > 0
    est = sa.align_to_photo(pts, mask, depth, K)
    err = np.linalg.norm(sa.apply(pts, est) - sa.apply(pts, gt), axis=1)
    assert np.median(err) < 0.04 * 1.7, np.median(err)     # within a few cm on a 1.7 m subject
    assert abs(est["scale"] / s_gt - 1) < 0.08
    # the back, which the photo never saw, ends up behind the front
    back = sa.apply(pts, est)[:, 2].max()
    assert back > np.median(depth[mask]) + 0.1


def test_depth_render_of_points():
    pts = np.stack(np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200), [3.0]), -1).reshape(-1, 3)
    K = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    d = sa.render_depth(pts, K, np.eye(4), 64, 64)
    assert np.allclose(d[16:48, 16:48], 3.0)
    assert (d[0] == 0).all()                                   # outside the square: empty


def look_at(eye, target):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, -1.0, 0])
    r /= np.linalg.norm(r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, np.cross(f, r), f, eye
    return m


def test_the_side_view_gets_the_subject_depth_instead_of_its_own_guess(tmp_path):
    v, f = figure()
    np.savez(tmp_path / "s.npz", vertices=v, faces=f, colors=np.full_like(v, 0.5))
    pts, _ = sa.sample_surface(v, f, 60000)
    rots = sa.candidate_rotations(24)
    gt = {"scale": 2.0, "R": rots[4 * 24], "t": np.array([0.0, 0.0, 2.5])}
    W, H = 320, 240
    K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]])
    photo = sa.render_depth(sa.apply(pts, gt), K, np.eye(4), W, H)
    side = look_at(np.array([2.5, 0, 2.5]), [0, 0, 2.5])            # 90° to the side, same distance
    truth = sa.render_depth(sa.apply(pts, gt), K, side, W, H, dilate=2)
    wrong = np.where(truth > 0, truth * 1.3, 0).astype(np.float32)   # a monocular guess, 30 % off
    priors = [photo.astype(np.float32), wrong]
    cams = [(K, np.eye(4), (W, H)), (K, side, (W, H))]
    align, world, cols = sa.place_subject(str(tmp_path / "s.npz"), photo > 0, photo, K, cams, priors, [False, True])
    assert np.array_equal(priors[0], photo.astype(np.float32))     # the photo keeps its own depth
    both = (truth > 0) & (priors[1] > 0)
    assert np.median(np.abs(priors[1][both] - truth[both]) / truth[both]) < 0.03
    assert align["replaced"][1] > 0.1 and world.shape[1] == 3 and cols.shape == world.shape
