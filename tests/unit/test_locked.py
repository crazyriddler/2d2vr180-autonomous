"""Real 3D v3 (rc56): the photo locked - scale check from the eyes and the diagnostic views."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))


def test_face_scale_always_brings_the_eyes_to_the_target_distance():
    """rc58 (owner: people always came out too big): the scale is set from the eyes every time, not only when
    they fall outside the adult range."""
    from sharp_depth_worker import face_scale

    assert face_scale([]) == 1.0
    assert face_scale([{"ipd": 0.063}]) == pytest.approx(1.0)
    assert face_scale([{"ipd": 0.068}]) == pytest.approx(0.063 / 0.068)       # 8 % too big → smaller
    assert face_scale([{"ipd": 0.068}], 0.056) == pytest.approx(0.056 / 0.068)  # a smaller target: smaller still
    assert face_scale([{"ipd": 0.5}]) == 0.5                                    # never more than 2x either way
    # several faces: the median decides
    assert face_scale([{"ipd": 0.050}, {"ipd": 0.070}, {"ipd": 0.072}]) == pytest.approx(0.063 / 0.070)


def test_people_shape_is_fused_inside_their_outline_at_the_scene_distance():
    """A 'face' whose scene depth is flat (as monocular depth flattens noses) gets the human model's relief,
    at the scene's own distance, blended over a few pixels inside the outline; outside nothing changes."""
    from human_geometry import crop_box, fuse_people, inner_weight

    h, w = 120, 100
    ys, xs = np.mgrid[0:h, 0:w]
    person = (xs - 50) ** 2 / 30 ** 2 + (ys - 60) ** 2 / 45 ** 2 < 1
    scene = np.where(person, 2.4, 5.0)                                   # flat person 2.4 m away
    nose = 0.04 * np.exp(-((xs - 50) ** 2 + (ys - 55) ** 2) / 60.0)
    human = np.where(person, 2.0 - nose, np.nan)                          # relief, but 2 m away
    fused, ratio = fuse_people(scene, np.ones((h, w), bool), human, person)
    assert ratio == pytest.approx(1.2, rel=0.01)                          # scene puts them 1.2x as far
    assert np.allclose(fused[~person], 5.0)
    relief = fused[55, 50] - fused[55, 32]
    assert relief == pytest.approx(-0.04 * 1.2, rel=0.15)                 # the nose sticks out (scaled)
    wgt = inner_weight(person, 6)
    assert wgt[~person].max() == 0 and wgt[60, 50] == 1.0 and 0 < wgt[60, 21] < 1
    box = crop_box(person)
    y0, y1, x0, x1 = box
    assert y0 <= 15 and y1 >= 105 and x0 <= 20 and x1 >= 80
    assert abs((x1 - x0) / (y1 - y0) - 0.75) < 0.05                       # the network's 3:4 aspect
    assert crop_box(np.zeros((50, 50), bool)) is None


def test_sheet_cameras_move_the_head_and_turn_around_the_subject():
    from twod2vr180.backends.locked import SHEET_VIEWS, sheet_cameras
    from twod2vr180.scene import Camera

    cams = sheet_cameras(Camera(1600, 1200, 1400, 1400, 800, 600), pivot=2.5, side=400)
    assert [c[0] for c in cams] == ["photo"] + [v[0] for v in SHEET_VIEWS]
    assert cams[0][1]["width"] == 400 and cams[0][1]["height"] == 300 and cams[0][1]["fx"] == pytest.approx(350)
    left = np.array(cams[1][1]["c2w"])
    assert left[0, 3] == pytest.approx(-0.15) and np.allclose(left[:3, :3], np.eye(3))
    turn = np.array(cams[4][1]["c2w"])                           # orbit: still looking at the pivot
    look = turn[:3, :3] @ [0, 0, 1.0]
    to_pivot = np.array([0, 0, 2.5]) - turn[:3, 3]
    assert np.allclose(look, to_pivot / np.linalg.norm(to_pivot), atol=1e-6)


def test_hole_depth_continues_the_background_behind_an_object():
    """A wall 4 m away seen past a box 2 m away: the hole uncovered next to the box must get the wall's
    depth, even though the monocular depth of the filled image has its own scale and offset."""
    from sharp_depth_worker import align_affine, dilate, hole_depth

    h, w = 120, 160
    known = np.full((h, w), 4.0)
    known[40:80, 50:90] = 2.0                                     # the box
    hole = np.zeros((h, w), bool)
    hole[40:80, 90:100] = True                                    # uncovered strip right of the box
    known[hole] = np.inf
    truth = np.where(hole, 4.0, known)
    mono = 0.5 * truth + 0.3                                      # what MoGe says: other scale and shift
    depth, valid, st = hole_depth(known, mono, hole, np.ones((h, w), bool), ring_px=6)
    assert valid.all()
    assert np.allclose(depth[hole], 4.0, atol=0.05)               # the wall, not the box
    assert np.allclose(depth[~hole], known[~hole])                # what exists stays exactly as rendered
    assert st["hole_fraction"] > 0 and st["ring_pixels"] > 100
    a, b = align_affine(np.array([1.0, 2, 3, 4] * 10), np.array([2.5, 4.5, 6.5, 8.5] * 10))
    assert a == pytest.approx(2.0) and b == pytest.approx(0.5)
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert dilate(m, 2).sum() == 25
