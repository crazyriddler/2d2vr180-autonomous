"""Real 3D v3 (rc56): the photo locked - scale check from the eyes and the diagnostic views."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))


def test_face_scale_only_corrects_clearly_wrong_depth():
    from sharp_depth_worker import face_scale

    assert face_scale([]) == 1.0
    assert face_scale([{"ipd": 0.063}]) == 1.0                    # inside the adult range: confirmed
    assert face_scale([{"ipd": 0.060}, {"ipd": 0.066}]) == 1.0
    s = face_scale([{"ipd": 0.084}])                              # depth 33 % too far → brought to 70 mm
    assert s == pytest.approx(0.070 / 0.084)
    s = face_scale([{"ipd": 0.040}])                              # too near → brought to 56 mm
    assert s == pytest.approx(0.056 / 0.040)
    assert face_scale([{"ipd": 0.5}]) == 0.5                      # never more than 2x either way
    # several faces: the median decides (one child or a poster does not)
    assert face_scale([{"ipd": 0.050}, {"ipd": 0.080}, {"ipd": 0.081}]) == pytest.approx(0.070 / 0.080)


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
