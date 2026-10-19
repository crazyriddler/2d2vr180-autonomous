"""Maximum-detail photos (workers/detail_geometry.py). Needs torch (present in the GPU runtime)."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))


def test_crops_cover_the_picture_and_own_it_exactly_once():
    import detail_geometry as dg

    boxes = dg.tile_boxes(3000, 2000, 2)
    count = np.zeros((2000, 3000), int)
    for crop, core in boxes:
        x0, y0, x1, y1 = core
        count[y0:y1, x0:x1] += 1
        assert crop[0] <= x0 and crop[2] >= x1 and crop[1] <= y0 and crop[3] >= y1
    assert (count == 1).all()
    own = sum(dg.owned(c, k).sum() for c, k in boxes)
    assert abs(own - 768 * 768 * 4 / 1.2 ** 2) / own < 0.02          # each crop keeps its core (crop = 1.2 x core)


def test_patch_fusion_keeps_the_whole_photo_shape_and_adds_the_crops_relief():
    import detail_geometry as dg

    h, w = 400, 600
    xs = np.arange(w)[None, :].repeat(h, 0).astype(float)
    truth = 3.0 + 0.4 * xs / w - 0.01 * np.sin(xs / 5)               # a slope with 1 cm grooves
    whole = 3.0 + 0.4 * xs / w                                       # the whole photo's depth: no grooves
    crops = []
    for crop, _ in dg.tile_boxes(w, h, 2):
        x0, y0, x1, y1 = crop
        crops.append((crop, 1.3 * truth[y0:y1, x0:x1] - 0.2))        # crops: right relief, wrong scale
    fused, stats = dg.patch_fusion(whole, crops, sigma=20, torch=torch)   # relief finer than sigma: from the crops
    inner = (slice(20, -20), slice(20, -20))
    assert np.abs(fused - truth)[inner].max() < 0.004                # grooves recovered, scale from the whole
    assert all(abs(s["scale"] - 1 / 1.3) < 0.02 for s in stats)


def test_edge_weight_is_low_at_depth_jumps():
    import detail_geometry as dg

    d = np.full((100, 100), 3.0)
    d[:, 50:] = 5.0
    w = dg.edge_weight(d, torch)
    assert w[50, 20] > 0.95 and w[50, 50] < 0.2 and w[50, 49] < 0.2


def test_very_large_photos_use_more_parts():
    """rc7: photos of 24 MP and up are worked at 4608 px in 3 x 3 parts (about twice the splats of 2 x 2)."""
    import detail_geometry as dg

    assert dg.work_side(4000) == 3072 and dg.work_side(6000) == 4608 and dg.work_side(8064) == 4608
    assert dg.tile_count(3072, 2304) == 2 and dg.tile_count(4608, 3456) == 3 and dg.tile_count(1536, 1152) == 1
    boxes = dg.tile_boxes(4608, 3456, 3)
    count = np.zeros((3456, 4608), np.uint8)
    for _, (x0, y0, x1, y1) in boxes:
        count[y0:y1, x0:x1] += 1
    assert len(boxes) == 9 and (count == 1).all()
