import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import fusion  # noqa: E402

K = np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
H, W = 96, 128


def yaw_c2w(deg, radius=5.0):
    """Camera orbiting the point (0, 0, radius) - like the generated shots."""
    a = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = np.array([[math.cos(-a), 0, math.sin(-a)], [0, 1, 0], [-math.sin(-a), 0, math.cos(-a)]])
    m[:3, 3] = [radius * math.sin(a), 0, radius * (1 - math.cos(a))]
    return m


def wall_depth(c2w, z_wall=5.0):
    """Camera-space depth of the plane z = z_wall (world) seen from c2w."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    d_cam = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
    d_w = d_cam @ c2w[:3, :3].T
    t = (z_wall - c2w[2, 3]) / d_w[..., 2]
    return np.where(t > 0, t, np.nan)


def view(c2w, generated, z=None):
    z = wall_depth(c2w) if z is None else z
    img = np.full((H, W, 3), 128, np.uint8)
    return fusion.View(z, K, c2w, img, np.isfinite(z), generated)


def test_photo_kept_whole_and_side_view_adds_only_new_wall():
    ref, side = view(np.eye(4), False), view(yaw_c2w(30), True)
    order = fusion.fusion_order([ref.c2w, side.c2w])
    assert order == [0, 1]
    fusion.fuse([ref, side], order)
    assert ref.keep.sum() == ref.valid.sum()
    kept = side.world(side.keep)
    assert len(kept) > 0
    # everything the side view adds lies outside the photo's field of view
    u, v, z, inb = ref.project(kept)
    assert inb.mean() < 0.05
    # and the side view drops most of what the photo already shows
    assert side.keep.sum() < 0.6 * side.valid.sum()


def test_points_floating_in_front_of_the_photo_are_dropped():
    ref = view(np.eye(4), False)
    c2w = yaw_c2w(10)
    side = view(c2w, True, z=wall_depth(c2w) * 0.6)    # inconsistent: everything much nearer
    fusion.fuse([ref, side], [0, 1])
    u, v, z, inb = ref.project(side.world(side.keep))
    assert inb.sum() == 0


def test_assemble_layout_and_provenance():
    ref, side = view(np.eye(4), False), view(yaw_c2w(40), True)
    fusion.fuse([ref, side], [0, 1])
    out, prov = fusion.assemble([ref, side])
    n = len(prov)
    assert out["means"].shape == (n, 3) and out["quats"].shape == (n, 4) and out["sh0"].shape == (n, 1, 3)
    assert set(np.unique(prov)) == {1, 2}
    assert np.allclose(np.linalg.norm(out["quats"], axis=1), 1, atol=1e-5)
    capped, prov2 = fusion.assemble([ref, side], max_gaussians=int(ref.keep.sum()) + 10)
    assert (prov2 == 2).sum() == 10 and (prov2 == 1).sum() == ref.keep.sum()


def test_align_scale_is_robust():
    rng = np.random.default_rng(0)
    z = rng.uniform(1, 5, (50, 50))
    ref = 2.5 * z
    ref[:5] = 100       # outliers
    assert abs(fusion.align_scale(z, ref, np.ones_like(z, bool)) - 2.5) < 1e-6


def test_small_depth_disagreement_counts_as_the_same_surface():
    ref = view(np.eye(4), False)
    c2w = yaw_c2w(15)
    side = view(c2w, True, z=wall_depth(c2w) * 1.06)   # 6 % further: same wall, imprecise depth
    fusion.fuse([ref, side], [0, 1])
    u, v, z, inb = ref.project(side.world(side.keep))
    assert inb.mean() < 0.05
