import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import seva_worker as sw  # noqa: E402


def test_explore_starts_at_the_photo_and_is_zero_mean():
    c2ws = sw.explore_c2ws(80)
    assert np.allclose(c2ws[0], np.eye(4))
    # Stable Virtual Camera rescales translations by the first camera's distance to the mean
    # camera position; a zero mean keeps the sway at its intended size.
    assert np.allclose(c2ws[:, :3, 3].mean(0), 0, atol=1e-9)
    for m in c2ws:
        assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
    yaws = np.degrees(np.arctan2(c2ws[:, 0, 2], c2ws[:, 2, 2]))
    assert yaws.max() > 60 and yaws.min() < -60


def test_intrinsics_follow_the_field_of_view():
    K = sw.intrinsics(90.0, 200, 100, 3)
    assert K.shape == (3, 3, 3) and np.isclose(K[0, 0, 0], 100.0) and np.isclose(K[0, 1, 2], 50)
