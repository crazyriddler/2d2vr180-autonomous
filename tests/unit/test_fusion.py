"""Real 3D fill geometry: synthetic splats of a box in front of a wall, seen by the photo and by moved views."""

import numpy as np
import pytest

from twod2vr.fusion import DepthBuffer, orbit_pose, quat_mul, rotmat_to_quat, rubber_sheet, transform_scene
from twod2vr.scene import GENERATIVE, Camera, GaussianScene, quat_to_rotmat

BOX_LO, BOX_HI = np.array([-0.3, -0.3, 1.7]), np.array([0.3, 0.3, 2.3])
WALL_Z = 4.0
W, H, F = 200, 150, 170.0


def _cast(c2w, w=W, h=H, f=F):
    """World hit points and colours of every pixel ray of a camera (box first, else the wall)."""
    ys, xs = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(xs + 0.5 - w / 2) / f, (ys + 0.5 - h / 2) / f, np.ones_like(xs, float)], -1).reshape(-1, 3)
    d = d_cam @ c2w[:3, :3].T
    o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (BOX_LO - o) / d, (BOX_HI - o) / d
    tn, tf = np.nanmax(np.minimum(t1, t2), 1), np.nanmin(np.maximum(t1, t2), 1)
    box = (tn <= tf) & (tf > 0)
    with np.errstate(divide="ignore"):
        tw = (WALL_Z - o[2]) / d[:, 2]
    t = np.where(box, tn, tw)
    X = o + d * t[:, None]
    col = np.tile([0.2, 0.3, 0.9], (len(X), 1))
    on_box = box.copy()
    front = on_box & (np.abs(X[:, 2] - BOX_LO[2]) < 1e-6)
    col[on_box] = [0.1, 0.8, 0.1]                     # sides and top: green
    col[front] = [0.9, 0.1, 0.1]                      # the face the photo sees: red
    # surfaces have texture, like real ones
    tex = 0.75 + 0.25 * np.sin(17 * X[:, 0] + 3 * X[:, 2]) * np.sin(13 * X[:, 1] + 5 * X[:, 2])
    col = col * np.nan_to_num(tex)[:, None]
    ok = np.isfinite(X).all(1) & (t > 0)
    return X[ok], col[ok]


def _sharp(c2w, scale, colour_gain=1.0):
    """What SHARP would output for a camera: its visible surfaces in its own frame, at its own scale."""
    X, col = _cast(c2w)
    w2c = np.linalg.inv(c2w)
    P = (X @ w2c[:3, :3].T + w2c[:3, 3]) * scale
    n = len(P)
    sc = GaussianScene(P.astype(np.float32), np.full((n, 3), 0.004 * scale, np.float32),
                       np.tile(np.float32([1, 0, 0, 0]), (n, 1)), np.full(n, 0.9, np.float32),
                       np.clip(col * colour_gain, 0, 1).astype(np.float32))
    return sc


def _orbit_c2w(az, el, pivot=2.0):
    s, R, t = orbit_pose(az, el, pivot)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, t
    return m


CAM = Camera(W, H, F, F, W / 2, H / 2)


def _surface_distance(P):
    """Distance of world points to the nearest true surface (box or wall)."""
    q = np.maximum(np.abs(P - (BOX_LO + BOX_HI) / 2) - (BOX_HI - BOX_LO) / 2, 0)
    outside = np.linalg.norm(q, axis=1)
    inside = np.minimum(np.abs(P - BOX_LO), np.abs(P - BOX_HI)).min(1)
    box = np.where(outside > 0, outside, inside)
    return np.minimum(box, np.abs(P[:, 2] - WALL_Z))


def test_quaternion_helpers_agree_with_rotation_matrices():
    rng = np.random.default_rng(1)
    for _ in range(5):
        A, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        A *= np.sign(np.linalg.det(A))
        q = rotmat_to_quat(A)
        assert np.allclose(quat_to_rotmat(q[None])[0], A, atol=1e-6)
        q2 = rng.normal(size=(1, 4))
        q2 /= np.linalg.norm(q2)
        assert np.allclose(quat_to_rotmat(quat_mul(q, q2))[0], A @ quat_to_rotmat(q2)[0], atol=1e-6)


def test_transform_scene_moves_positions_sizes_and_orientations():
    sc = _sharp(np.eye(4), 1.0)
    R = quat_to_rotmat(rotmat_to_quat(orbit_pose(40, 10, 2)[1])[None])[0]
    out = transform_scene(sc, 2.0, R, np.array([1.0, 0, 0]))
    assert np.allclose(out.means, 2 * sc.means @ R.T + [1, 0, 0], atol=1e-4)
    assert np.allclose(out.scales, 2 * sc.scales)
    assert np.allclose(quat_to_rotmat(out.quats[:1])[0], R, atol=1e-5)


def test_rubber_sheet_splats_across_depth_edges_are_found():
    sc = _sharp(np.eye(4), 1.0)
    # a stretched splat half-way between the box edge and the wall, on the box's silhouette
    extra = np.array([[(0.3 / 1.7 + 0.001) * 3.0, 0.0, 3.0]], np.float32)
    sheet = GaussianScene(np.concatenate([sc.means, extra]), np.concatenate([sc.scales, sc.scales[:1]]),
                          np.concatenate([sc.quats, sc.quats[:1]]), np.concatenate([sc.opacity, [0.9]]).astype(np.float32),
                          np.concatenate([sc.colors, sc.colors[:1]]))
    flags = rubber_sheet(sheet, CAM)
    assert flags[-1]
    assert flags[:-1].mean() < 0.01


@pytest.mark.parametrize("az", [45, -45])
def test_orbit_pose_looks_at_the_pivot(az):
    s, R, t = orbit_pose(az, 20, 2.0)
    centre_ray = R @ np.array([0, 0, 1.0])
    pivot = np.array([0, 0, 2.0])
    to_pivot = (pivot - t) / np.linalg.norm(pivot - t)
    assert np.allclose(centre_ray, to_pivot, atol=1e-6)
    assert np.isclose(np.linalg.norm(pivot - t), 2.0)
    assert t[1] < 0                                              # + elevation: the camera is above


def test_holes_are_what_the_photo_never_saw_and_fills_add_only_them():
    """rc57: a head move to the right uncovers the wall behind the box's right edge. The render of the
    photo's splats there is empty; a fill made from that view (here: the true surfaces) adds exactly that
    strip and the box's side, and nothing the photo already shows."""
    from twod2vr.fusion import add_fills, unseen_by_photo
    from twod2vr.render import render_pinhole

    base = _sharp(np.eye(4), 1.0)
    photo_buf = DepthBuffer(CAM).fill(base.means)
    c2w = np.eye(4)
    c2w[0, 3] = 0.35                                              # head 35 cm to the right
    r = render_pinhole(base, c2w, W, H, F, F, W / 2, H / 2, crack_fill=False)
    hole = unseen_by_photo(r.depth, r.covered.astype(float), CAM, c2w, photo_buf)
    assert 0.005 < hole.mean() < 0.2
    # the hole is right of the box (the wall the box hid), not on the box or the open wall
    ys, xs = np.nonzero(hole[H // 3: 2 * H // 3])
    box_right = (0.3 - 0.35) / 1.7 * F + W / 2
    assert np.median(xs) > box_right - 5
    fill_scene = _sharp(c2w, 1.0)                                 # what the view really sees
    merged, added, rep = add_fills(base, CAM, [(fill_scene, CAM, c2w, hole, "right 35 cm")])
    assert len(added) > 50 and len(merged) == len(base) - rep[-1]["replaced_by_new_content"] + len(added)
    assert (added.provenance == GENERATIVE).all()
    assert np.median(_surface_distance(added.means)) < 0.01
    u, v, z, inside = photo_buf.project(added.means)
    seen = inside & (np.abs(z / photo_buf.depth[v, u] - 1) < 0.02)
    assert seen.mean() < 0.05                                     # nothing the photo shows is duplicated
    assert rep[0]["added"] == len(added)


def test_rendered_holes_include_guesses_behind_the_photo_surfaces():
    """A splat behind the box (a 'second layer' guess) is drawn in a moved view, but the photo never saw
    it: its pixels still count as holes, and new content there replaces the guess."""
    from twod2vr.fusion import add_fills, unseen_by_photo
    from twod2vr.render import render_pinhole

    base = _sharp(np.eye(4), 1.0)
    n = 400
    rng = np.random.default_rng(0)
    guess = np.stack([rng.uniform(0.3, 0.5, n), rng.uniform(-0.25, 0.25, n), np.full(n, 3.0)], 1)
    g = GaussianScene(guess.astype(np.float32), np.full((n, 3), 0.01, np.float32),
                      np.tile(np.float32([1, 0, 0, 0]), (n, 1)), np.full(n, 0.9, np.float32),
                      np.tile(np.float32([1, 0, 1]), (n, 1)))
    both = GaussianScene(*(np.concatenate([getattr(base, k), getattr(g, k)])
                           for k in ("means", "scales", "quats", "opacity", "colors")))
    photo_buf = DepthBuffer(CAM).fill(both.means)
    c2w = np.eye(4)
    c2w[0, 3] = 0.35
    r = render_pinhole(both, c2w, W, H, F, F, W / 2, H / 2, crack_fill=False)
    hole = unseen_by_photo(r.depth, r.covered.astype(float), CAM, c2w, photo_buf)
    magenta = (r.rgb[..., 0] > 200) & (r.rgb[..., 1] < 60) & (r.rgb[..., 2] > 200)
    assert magenta.sum() > 20 and hole[magenta].mean() > 0.9
    merged, added, rep = add_fills(both, CAM, [(_sharp(c2w, 1.0), CAM, c2w, hole, "right")])
    assert rep[-1]["guessed_behind_surfaces"] >= n * 0.9 and rep[-1]["replaced_by_new_content"] > 0
