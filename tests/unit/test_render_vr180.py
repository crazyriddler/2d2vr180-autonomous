import numpy as np
import pytest

from twod2vr180.render import orbit_c2w, render_equirect180, render_pinhole
from twod2vr180.rgbd import pointmap_to_gaussians, pointmap_to_obj
from twod2vr180.scene import Camera
from twod2vr180.vr180 import StereoOptions, render_stereo, save_stereo_image


def plane_scene(w=160, h=120, f=150.0, box_depth=1.5, far=3.0):
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.full((h, w), far)
    z[h // 3:2 * h // 3, w // 3:2 * w // 3] = box_depth
    pts = np.stack([(xs + .5 - w / 2) / f * z, (ys + .5 - h / 2) / f * z, z], -1)
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = xs * 255 // w
    img[..., 1] = ys * 255 // h
    img[h // 3:2 * h // 3, w // 3:2 * w // 3] = 255
    cam = Camera(w, h, f, f, w / 2, h / 2)
    return pts, img, cam


def white_centroid_x(im):
    m = im.min(-1) > 245
    return np.nonzero(m)[1].mean()


def test_source_view_fidelity():
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    r = render_pinhole(sc, np.eye(4), cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
    cov = r.covered | r.filled
    assert cov.mean() > 0.97
    mae = np.abs(r.rgb.astype(int) - img)[cov].mean()
    assert mae < 4


def test_no_vertical_or_horizontal_flip():
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    for proj in ("flat", "equirect180"):
        fr = render_stereo(sc, StereoOptions(projection=proj, eye_resolution=240, layout="sbs"))
        left = fr.left.rgb.astype(int)
        cov = fr.left.covered
        ys, xs = np.nonzero(cov)
        top = left[ys.min() + 2, int(xs.mean())]
        bottom = left[ys.max() - 2, int(xs.mean())]
        assert bottom[1] > top[1] + 100, proj  # green grows downwards
        lft = left[int(ys.mean()), xs.min() + 2]
        rgt = left[int(ys.mean()), xs.max() - 2]
        assert rgt[0] > lft[0] + 100, proj  # red grows to the right


@pytest.mark.parametrize("proj", ["flat", "equirect180"])
def test_eye_order_near_object_disparity(proj):
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    fr = render_stereo(sc, StereoOptions(projection=proj, eye_resolution=240, eye_separation_m=0.1))
    # the near white box appears further right in the LEFT eye than in the right eye
    assert white_centroid_x(fr.left.rgb) > white_centroid_x(fr.right.rgb) + 0.5
    # layout: left eye occupies the left half (SBS) / top half (TB)
    np.testing.assert_array_equal(fr.image[:, : fr.left.rgb.shape[1]], fr.left.rgb)
    tb = render_stereo(sc, StereoOptions(projection=proj, eye_resolution=240, eye_separation_m=0.1, layout="tb"))
    np.testing.assert_array_equal(tb.image[: tb.left.rgb.shape[0]], tb.left.rgb)


def test_equirect_geometry_and_honesty(tmp_path):
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    fr = render_stereo(sc, StereoOptions(eye_resolution=256, eye_separation_m=0.0))
    assert fr.image.shape == (256, 512, 3)
    md = fr.metadata
    hfov = 2 * np.degrees(np.arctan(cam.width / 2 / cam.fx))
    assert md["content_fov_deg"]["horizontal_deg"] == pytest.approx(hfov, abs=3)
    assert md["is_full_vr180"] is False and "not a full hemispherical" in md["honesty_note"]
    # content centred: the image centre is covered, the far edges are not
    assert fr.left.covered[128, 128] and not fr.left.covered[128, 2]
    saved = save_stereo_image(fr, StereoOptions(), tmp_path, "t")
    assert saved["image"].endswith("t_180_LR.jpg")


def test_equirect_point_directions():
    # three tiny splats at known angles
    from twod2vr180.scene import GaussianScene

    dirs = np.array([[0, 0, 1], [1, 0, 1], [0, -1, 1]], np.float64)  # centre, 45° right, 45° up
    means = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * 5).astype(np.float32)
    sc = GaussianScene(means, np.full((3, 3), 0.01, np.float32), np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
                       np.ones(3, np.float32), np.eye(3, dtype=np.float32))
    r = render_equirect180(sc, np.eye(4), 360, 360, crack_fill=False)
    ys, xs = np.nonzero(r.covered)
    got = sorted(zip(xs.tolist(), ys.tolist()))
    assert (180, 180) in got
    assert (270, 180) in got   # lon +45° → u = 0.75 W
    assert (180, 90) in got    # lat +45° → v = 0.25 H


def test_orbit_view_reveals_unknown_not_invented():
    pts, img, cam = plane_scene()
    sc = pointmap_to_gaussians(pts, img, None, cam)
    c2w = orbit_c2w(np.array([0, 0, 3.0]), 3.0, 35, 0)
    r = render_pinhole(sc, c2w, 160, 120, 150, 150, 80, 60)
    st = r.coverage_stats()
    assert st["unknown_fraction"] > 0.02
    assert st["generative_fraction"] == 0


def test_obj_export_drops_discontinuities(tmp_path):
    pts, img, cam = plane_scene()
    info = pointmap_to_obj(pts, img, None, tmp_path / "m.obj")
    assert info["faces"] > 0 and info["dropped_discontinuity_quads"] > 0
    txt = (tmp_path / "m.obj").read_text()
    assert "mtllib m.mtl" in txt and (tmp_path / "m_texture.png").exists()
    nv = sum(1 for l in txt.splitlines() if l.startswith("v "))
    nf = sum(1 for l in txt.splitlines() if l.startswith("f "))
    assert nv == info["vertices"] and nf == info["faces"]
    idx = [int(t.split("/")[0]) for l in txt.splitlines() if l.startswith("f ") for t in l.split()[1:]]
    assert min(idx) >= 1 and max(idx) <= nv


def test_invalid_stereo_options():
    with pytest.raises(ValueError):
        StereoOptions(layout="xy").validate()
