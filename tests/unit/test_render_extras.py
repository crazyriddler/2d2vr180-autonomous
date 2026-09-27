import numpy as np
import pytest
from PIL import Image

from test_render_vr180 import plane_scene, white_centroid_x
from twod2vr180.render import fill_disocclusions, render_pinhole
from twod2vr180.rgbd import disparity_to_points, pointmap_to_gaussians
from twod2vr180.vr180 import StereoOptions, assemble_gpu, finalize_stereo, gpu_views, plan_stereo, render_stereo


def fake_gpu_render(scene, views):
    """Stand-in for workers/render_worker.py: premultiplied RGB + alpha + depth."""
    out = []
    for v in views:
        r = render_pinhole(scene, v["c2w"], v["width"], v["height"], v["fx"], v["fy"], v["cx"], v["cy"],
                           crack_fill=False)
        a = r.covered.astype(np.float32)
        out.append({"rgb": (r.rgb.astype(np.float32) * a[..., None]).astype(np.uint8), "alpha": a,
                    "depth": np.where(r.covered, r.depth, 0)})
    return out


@pytest.mark.parametrize("proj", ["equirect180", "flat"])
def test_gpu_view_assembly_matches_cpu(proj):
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    opts = StereoOptions(projection=proj, eye_resolution=256, eye_separation_m=0.1, fill_holes=False)
    plan = plan_stereo(sc, opts)
    views = gpu_views(plan)
    assert len(views) == (10 if proj == "equirect180" else 2)
    left, right = assemble_gpu(plan, fake_gpu_render(sc, views))
    gpu = finalize_stereo(sc, opts, plan, left, right, renderer="test")
    cpu = render_stereo(sc, opts)
    assert gpu.image.shape == cpu.image.shape
    both = gpu.left.covered & cpu.left.covered
    assert both.sum() > 0.9 * cpu.left.covered.sum()
    assert np.abs(gpu.left.rgb[both].astype(int) - cpu.left.rgb[both]).mean() < 12
    # eye order survives the cube-map path
    assert white_centroid_x(gpu.left.rgb) > white_centroid_x(gpu.right.rgb)


def test_hole_filling_uses_background_and_is_reported():
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    raw = render_stereo(sc, StereoOptions(projection="flat", eye_resolution=240, eye_separation_m=0.2,
                                          fill_holes=False))
    filled = render_stereo(sc, StereoOptions(projection="flat", eye_resolution=240, eye_separation_m=0.2))
    st_raw, st = raw.metadata["coverage"]["left"], filled.metadata["coverage"]["left"]
    assert st_raw["unknown_fraction"] > 0.005
    assert st["hole_filled_fraction"] > 0 and st["unknown_fraction"] < st_raw["unknown_fraction"]
    holes = filled.left.hole_filled
    # filled pixels take the background colour, never the white foreground box
    assert (filled.left.rgb[holes].min(-1) < 245).all()


def test_hole_filling_leaves_outside_fov_unknown():
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    fr = render_stereo(sc, StereoOptions(eye_resolution=256))
    assert fr.metadata["coverage"]["left"]["unknown_fraction"] > 0.8
    assert not fr.left.covered[128, 2] and not fr.left.hole_filled[128, 2]


def test_disparity_lift_is_ordered_and_bounded():
    disp = np.tile(np.linspace(0, 1, 64), (48, 1)).astype(np.float32)  # right = nearer
    pts, cam = disparity_to_points(disp, hfov_deg=60, near_m=1, far_m=10)
    z = pts[..., 2]
    assert z[:, -1].mean() < z[:, 0].mean()
    assert 0.9 < z.min() and z.max() < 10.5
    assert cam.hfov_deg == pytest.approx(60, abs=0.1)


def test_exif_hfov(tmp_path):
    from twod2vr180.media import exif_hfov_deg

    im = Image.new("RGB", (400, 300))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26  # 26 mm equivalent (typical phone main camera)
    p = tmp_path / "e.jpg"
    im.save(p, exif=exif)
    assert exif_hfov_deg(p) == pytest.approx(69.4, abs=0.5)
    q = tmp_path / "n.jpg"
    im.save(q)
    assert exif_hfov_deg(q) is None


def test_interpolate_poses_hits_keyframes():
    from twod2vr180.render import interpolate_poses, orbit_c2w

    keys = [orbit_c2w(np.array([0, 0, 3.0]), 3.0, a, 5) for a in (-20, 0, 25)]
    out = interpolate_poses(keys, 9)
    assert len(out) == 9
    for k, idx in ((0, 0), (1, 4), (2, 8)):
        np.testing.assert_allclose(out[idx], keys[k], atol=1e-6)
    for m in out:
        np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-6)
