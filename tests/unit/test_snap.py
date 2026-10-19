"""Splat videos: people's splats moved to their measured 3D shape (workers/sharp_depth_worker.snap_to_depth).
Needs torch (present in the GPU runtime, optional here)."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))


def test_people_splats_follow_the_measured_relief():
    import sharp_depth_worker as sd

    g = sd.SHARP_GRID
    h, w = 64, 48
    flat = torch.full((g, g), 2.0)                       # SHARP put the person flat at 2 m

    class G:
        pass

    gs = G()
    ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    m0 = torch.stack([(xs - g / 2) / g * flat, (ys - g / 2) / g * flat, flat], -1).reshape(-1, 3)
    gs.mean_vectors = torch.cat([m0, m0 * 1.2])[None]
    gs.singular_values = torch.full((1, 2 * g * g, 3), 0.01)
    depth = np.full((h, w), 2.0)
    depth[24:40, 18:30] = 1.9                            # a 10 cm relief the human model measured
    weight = np.zeros((h, w))
    weight[8:56, 8:40] = 1.0                             # the person
    means, scales, k = sd.snap_to_depth(gs, depth, weight, torch)
    z = means[: g * g, 2].reshape(g, g)
    assert abs(float(z[g // 2, g // 2]) - 1.9) < 0.01     # the relief is there
    assert abs(float(z[5, 5]) - 2.0) < 1e-6               # outside the person: SHARP's placement
    back = means[g * g:, 2].reshape(g, g)
    assert abs(float(back[g // 2, g // 2]) - 1.9 * 1.2) < 0.02   # the splat behind keeps its distance ratio
    assert abs(float(scales[g * g // 2 + g // 2, 0]) - 0.01 * 1.9 / 2.0) < 1e-3   # same size on screen
