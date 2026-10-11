"""4D video reconstruction: what moves, and the per-frame splats of the moving parts (numpy parts of
workers/video4d_worker.py; the GPU parts run only on the owner's machine)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import video4d_worker as v4  # noqa: E402

H, W = 90, 160
K = np.array([[120.0, 0, W / 2], [0, 120.0, H / 2], [0, 0, 1]])


def texture(seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (H // 6 + 1, W // 6 + 1, 3)).astype(np.uint8)
    return np.repeat(np.repeat(t, 6, 0), 6, 1)[:H, :W]


def frame(c2w, box_x, box_z=2.0, wall_z=5.0, tex=None):
    """A textured wall at z = wall_z and a red box (0.5 m wide) at x = box_x, z = box_z, seen from c2w."""
    tex = texture() if tex is None else tex
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
    dw = d @ c2w[:3, :3].T
    o = c2w[:3, 3]
    t_wall = (wall_z - o[2]) / dw[..., 2]
    hit = o + dw * t_wall[..., None]
    u = np.clip(((hit[..., 0] + 3.33) / 6.67 * W).astype(int), 0, W - 1)    # texture fixed to the wall
    v = np.clip(((hit[..., 1] + 1.9) / 3.75 * H).astype(int), 0, H - 1)
    img = tex[v, u].copy()
    z = t_wall.copy()
    t_box = (box_z - o[2]) / dw[..., 2]
    p = o + dw * t_box[..., None]
    inbox = (np.abs(p[..., 0] - box_x) < 0.25) & (np.abs(p[..., 1]) < 0.4)
    img[inbox] = [220, 30, 30]
    z[inbox] = t_box[inbox]       # depth along the optical axis == t for this ray parametrisation
    return img, z.astype(np.float32), inbox


def shift_c2w(x):
    m = np.eye(4)
    m[0, 3] = x
    return m


def test_still_pixels_vote_still_and_the_box_votes_moving():
    i0, z0, box0 = frame(np.eye(4), -0.6)
    i1, z1, box1 = frame(np.eye(4), 0.4)
    votes = v4.motion_votes(i0, z0, K, np.eye(4), i1, z1, K, np.eye(4))
    inner = box0 & ~v4.depth_edges(z0)                # on depth edges monocular depth says nothing
    assert (votes[inner] == v4.FREE_SPACE).mean() > 0.99      # the box left: the wall is there now
    wall = ~box0 & ~box1
    assert (votes[wall] == v4.STILL).mean() > 0.9
    assert (votes[~box0 & box1] == v4.NO_EVIDENCE).mean() > 0.9   # wall hidden behind the box then


@pytest.mark.parametrize("moving_camera", [False, True])
def test_moving_masks_find_the_box(moving_camera):
    pytest.importorskip("scipy")                      # the worker's runtime has it (mask clean-up)
    n, fps = 16, 4.0
    c2ws = [shift_c2w(0.04 * k if moving_camera else 0.0) for k in range(n)]
    frames = [frame(c2ws[k], -1.2 + 0.16 * k) for k in range(n)]
    imgs, depths, boxes = zip(*frames)
    masks = v4.moving_masks(list(imgs), list(depths), [K] * n, c2ws, fps, (0.5, 1.5))
    seen = np.zeros((H, W), bool)
    seen[:, 6:-6] = True          # points at the very edge of the first frames are seen by no other frame
    for k in range(n):
        assert masks[k][boxes[k] & seen].mean() > 0.97, k                  # the box is found ...
        grown = np.zeros_like(boxes[k])
        ys, xs = np.nonzero(boxes[k])
        grown[max(0, ys.min() - 6):ys.max() + 7, max(0, xs.min() - 6):xs.max() + 7] = True
        assert masks[k][~grown].mean() < 0.02, k                           # ... and little else


def test_a_person_standing_still_for_a_while_is_still_found_moving():
    """Neighbours 0.5 s away agree (they stood still), the ones 1.5 s away do not: moving wins."""
    pytest.importorskip("scipy")
    n, fps = 13, 2.0
    xs = [-1.0] * 2 + [0.0] * 9 + [1.0] * 2           # still from frame 2 to 10, moving before and after
    frames = [frame(np.eye(4), x) for x in xs]
    imgs, depths, boxes = zip(*frames)
    masks = v4.moving_masks(list(imgs), list(depths), [K] * n, [np.eye(4)] * n, fps, (0.5, 1.5, 3.0))
    assert masks[6][boxes[6]].mean() > 0.9


def test_camera_is_fixed():
    assert v4.camera_is_fixed([np.eye(4), shift_c2w(0.004)], depth_median=2.0)
    assert not v4.camera_is_fixed([np.eye(4), shift_c2w(0.1)], depth_median=2.0)
    r = np.eye(4)
    a = math.radians(3)
    r[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    assert not v4.camera_is_fixed([np.eye(4), r], depth_median=2.0)


def test_rotmat_to_quat_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert abs(abs(float(np.dot(v4.rotmat_to_quat(R), q))) - 1) < 1e-9


def test_frame_splats_lie_on_the_moving_surface_in_world_coordinates():
    c2w = shift_c2w(0.3)
    img, z, box = frame(c2w, 0.5)
    sp = v4.frame_splats(img, z, box, K, c2w, budget=150)
    assert sp is not None and len(sp["means"]) <= 260 and sp["stride"] >= 2
    assert np.allclose(sp["means"][:, 2], 2.0, atol=1e-4)                  # on the box face, in world z
    assert np.abs(sp["means"][:, 0] - 0.5).max() <= 0.26                   # inside the box, world x
    assert (sp["colors"][:, 0] > 0.8).all()
    px = 2.0 / K[0, 0]                                                       # one pixel at 2 m
    assert np.allclose(sp["scales"][:, 0], 0.75 * sp["stride"] * px, rtol=1e-4)
    assert v4.frame_splats(img, z, np.zeros_like(box), K, c2w, 100) is None


def test_gpu_renderer_draws_the_moving_splats_of_the_requested_time():
    """workers/render_worker.dynamic_at: what a 4D VR180 frame adds to the background."""
    torch = pytest.importorskip("torch")
    import render_worker as rw

    from twod2vr180.scene import GaussianScene
    from twod2vr180.scene4d import DynamicSplats, per_frame_splats

    frames = 4
    parts = []
    for k in range(frames):
        m = np.array([[0.1 * k, 0, 2.0]], np.float32)
        parts.append(per_frame_splats(GaussianScene(m, np.full((1, 3), 0.01, np.float32),
                                                    np.float32([[1, 0, 0, 0]]), np.ones(1, np.float32),
                                                    np.float32([[1, 0, 0]])), k, frames))
    path = Path(__file__).parent / "_dyn_tmp.npz"
    DynamicSplats.concat(parts).save(path)
    try:
        with np.load(path) as d:
            dyn = {k: d[k].astype(np.float32) for k in d.files if k != "provenance"}
    finally:
        path.unlink()
    for k in range(frames):
        means, quats, scales, op, sh = rw.dynamic_at(dyn, (k + 0.5) / frames, 4, torch, "cpu")
        assert means.shape == (1, 3) and abs(float(means[0, 0]) - 0.1 * k) < 1e-6
        assert sh.shape == (1, 4, 3) and float(sh[0, 0, 0]) > 1.5 and float(op[0]) > 0.99
