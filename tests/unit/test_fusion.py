"""Scene geometry helpers (fusion.py): quaternions, moving a splat, orbit poses, depth buffers."""

import numpy as np
import pytest

from twod2vr.fusion import DepthBuffer, orbit_pose, quat_mul, rotmat_to_quat, transform_scene
from twod2vr.scene import Camera, GaussianScene, quat_to_rotmat

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


CAM = Camera(W, H, F, F, W / 2, H / 2)


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


@pytest.mark.parametrize("az", [45, -45])
def test_orbit_pose_looks_at_the_pivot(az):
    s, R, t = orbit_pose(az, 20, 2.0)
    centre_ray = R @ np.array([0, 0, 1.0])
    pivot = np.array([0, 0, 2.0])
    to_pivot = (pivot - t) / np.linalg.norm(pivot - t)
    assert np.allclose(centre_ray, to_pivot, atol=1e-6)
    assert np.isclose(np.linalg.norm(pivot - t), 2.0)
    assert t[1] < 0                                              # + elevation: the camera is above


def test_depth_buffer_sees_the_nearest_surface():
    """What views_sheet uses to find the subject's distance (backends/locked.pivot_depth)."""
    sc = _sharp(np.eye(4), 1.0)
    buf = DepthBuffer(CAM, 128).fill(sc.means, sc.opacity > 0.1)
    c = buf.depth[buf.h // 2, buf.w // 2]
    assert abs(c - BOX_LO[2]) < 0.05                           # the box's face, not the wall behind it
    assert np.nanmax(buf.depth[np.isfinite(buf.depth)]) == pytest.approx(WALL_Z, abs=0.1)
