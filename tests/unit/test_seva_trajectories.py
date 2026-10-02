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


def test_progress_bar_reports_diffusion_steps(capsys):
    sw.ProgressBar.passes = 0
    for _ in sw.ProgressBar(enumerate(range(25)), total=25, leave=False):      # chunk loop (no description)
        for _ in sw.ProgressBar(range(2), total=2, desc="Sampling", leave=False):
            pass
        break
    out = capsys.readouterr().out
    assert "pass 1/2, chunk 1/25, step 2/2" in out and "s/step" in out


def test_short_sides_fit_the_pixel_budget():
    assert sw.short_sides(576, 1.0, 576 * 576) == [576, 512, 448]
    s = sw.short_sides(576, 4 / 3, 576 * 576)
    assert s[0] == 448 and all(x % 64 == 0 for x in s)          # 597x448 fits, 682x512 does not
    assert sw.short_sides(576, 16 / 9, 576 * 576)[0] == 384
