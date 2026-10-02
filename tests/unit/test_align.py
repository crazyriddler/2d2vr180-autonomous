import numpy as np

from twod2vr180.align import estimate_up, level_c2w, quat_xyzw_to_rotmat, viewer_transform
from twod2vr180.scene import Camera, GaussianScene


def _rx(a):
    a = np.radians(a)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rz(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _room(rcw):
    rng = np.random.default_rng(0)
    floor = np.stack([rng.uniform(-3, 3, 40000), np.full(40000, 1.5), rng.uniform(0.5, 6, 40000)], 1)
    wall = np.stack([rng.uniform(-3, 3, 30000), rng.uniform(-2, 1.5, 30000), np.full(30000, 6.0)], 1)
    pc = np.concatenate([floor, wall]) @ rcw      # world (level, OpenCV axes) → tilted camera
    n = len(pc)
    return GaussianScene(pc.astype(np.float32), np.full((n, 3), 0.01, np.float32),
                         np.tile(np.float32([1, 0, 0, 0]), (n, 1)), np.ones(n, np.float32),
                         np.zeros((n, 3), np.float32), cameras=[Camera(640, 480, 500, 500, 320, 240)])


def test_up_from_floor_of_pitched_and_rolled_camera():
    rcw = _rz(8) @ _rx(-20)
    est = estimate_up(_room(rcw))
    true_up = rcw.T @ np.array([0, -1.0, 0])
    assert est.method == "geometry"
    assert np.degrees(np.arccos(np.clip(est.up @ true_up, -1, 1))) < 1.0


def test_level_camera_keeps_position_and_heading():
    rcw = _rz(8) @ _rx(-20)
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    up = rcw.T @ np.array([0, -1.0, 0])
    L = level_c2w(c2w, up)
    assert np.allclose(L[:3, 3], [1, 2, 3])
    assert np.isclose(np.linalg.det(L[:3, :3]), 1.0)
    assert np.allclose(-L[:3, 1], up / np.linalg.norm(up))          # camera up == gravity up
    assert L[:3, 2] @ c2w[:3, 2] > 0.9                              # same heading


def test_viewer_transform_maps_gravity_to_plus_y_and_camera_to_origin():
    rcw = _rz(-5) @ _rx(12)
    sc = _room(rcw)
    xf = viewer_transform(sc)
    R = quat_xyzw_to_rotmat(xf["quaternion"])
    up = np.asarray(xf["up"]["up"])
    assert np.allclose(R @ up, [0, 1, 0], atol=0.02)
    assert np.allclose(xf["position"], 0, atol=1e-9)


def test_featureless_scene_falls_back_to_camera_up():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(5000, 3)).astype(np.float32) + [0, 0, 5]
    n = len(pts)
    sc = GaussianScene(pts, np.full((n, 3), 0.01, np.float32), np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                       np.ones(n, np.float32), np.zeros((n, 3), np.float32))
    est = estimate_up(sc)
    assert est.method == "camera" and np.allclose(est.up, [0, -1, 0])
