import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import multiview_worker as mv  # noqa: E402

H, W = 120, 518
K = np.array([[400.0, 0, W / 2], [0, 400.0, H / 2], [0, 0, 1]])


def yaw_c2w(deg, radius=5.0):
    a = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = np.array([[math.cos(-a), 0, math.sin(-a)], [0, 1, 0], [-math.sin(-a), 0, math.cos(-a)]])
    m[:3, 3] = [radius * math.sin(a), 0, radius * (1 - math.cos(a))]
    return m


def plane_depth(c2w, z_wall=5.0):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
    dw = d @ c2w[:3, :3].T
    return (z_wall - c2w[2, 3]) / dw[..., 2]


def render(texture, c2w):
    """The plane z=5 textured with the reference image, seen from c2w."""
    z = plane_depth(c2w)
    pts = mv.unproject(z, K, c2w)
    u = np.clip(np.floor(K[0, 0] * pts[..., 0] / pts[..., 2] + K[0, 2]).astype(int), 0, W - 1)
    v = np.clip(np.floor(K[1, 1] * pts[..., 1] / pts[..., 2] + K[1, 2]).astype(int), 0, H - 1)
    return texture[v, u]


def test_rigid_candidate_beats_a_changed_one(monkeypatch):
    rng = np.random.default_rng(0)
    tex = np.repeat(np.repeat(rng.random((H // 6, W // 7 + 1, 3)), 6, 0), 7, 1)[:H, :W].astype(np.float32)
    cam = yaw_c2w(12)
    rigid = render(tex, cam)
    changed = rigid.copy()
    changed[30:90, 200:330] = rng.random((60, 130, 3))       # e.g. the head turned
    w2c = np.linalg.inv(cam)[:3]

    def fake_vggt(model, batch, dtype, torch):
        return {"w2c": np.stack([np.eye(4)[:3], w2c]), "K": np.stack([K, K]),
                "depth": np.stack([plane_depth(np.eye(4)), plane_depth(cam)]), "conf": np.ones((2, H, W)),
                "pads": [0, 0]}

    monkeypatch.setattr(mv, "run_vggt", fake_vggt)
    ref = (tex, {})
    s_rigid, err_rigid, ang, ov = mv.rigidity_score(None, ref, (rigid, {}), None, None, target_deg=12)
    s_changed, err_changed, _, _ = mv.rigidity_score(None, ref, (changed, {}), None, None, target_deg=12)
    assert abs(ang - 12) < 0.5 and ov > 0.3
    assert err_rigid < 0.02 and s_changed > s_rigid
    # a candidate that did not move at all is penalised when a move was asked for
    def still(model, batch, dtype, torch):
        out = fake_vggt(model, batch, dtype, torch)
        out["w2c"] = np.stack([np.eye(4)[:3], np.eye(4)[:3]])
        out["depth"] = np.stack([plane_depth(np.eye(4))] * 2)
        return out

    monkeypatch.setattr(mv, "run_vggt", still)
    s_still, _, ang0, _ = mv.rigidity_score(None, ref, (tex, {}), None, None, target_deg=45)
    assert ang0 < 0.5 and s_still > 0.05


def test_consistency_map_flags_only_the_changed_region():
    rng = np.random.default_rng(1)
    tex = (np.repeat(np.repeat(rng.random((H // 6, W // 7 + 1, 3)), 6, 0), 7, 1)[:H, :W] * 255).astype(np.uint8)
    cam = yaw_c2w(10)
    view = (render(tex.astype(np.float32) / 255, cam) * 255).astype(np.uint8)
    view = np.clip(view.astype(int) + 12, 0, 255).astype(np.uint8)       # a little brighter overall
    changed = view.copy()
    changed[30:90, 200:330] = (rng.random((60, 130, 3)) * 255).astype(np.uint8)
    d0, d1 = plane_depth(np.eye(4)), plane_depth(cam)
    conf_ok = mv.consistency_map(tex, d0, K, np.eye(4), view, d1, K, cam)
    conf = mv.consistency_map(tex, d0, K, np.eye(4), changed, d1, K, cam)
    assert conf_ok.shape == (H, W) and np.median(conf_ok) > 0.9          # brightness offset is ignored
    assert conf[40:80, 215:315].mean() < 0.4                               # the changed region
    assert conf[:, :150].mean() > 0.85                                      # the rest stays trusted


def test_field_of_view_from_exif(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (400, 300))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26                     # a typical phone main camera
    img.save(tmp_path / "a.jpg", exif=exif)
    fov = mv.exif_fov_x(str(tmp_path / "a.jpg"))
    assert fov is not None and 65 < fov < 70             # ~67° across the long side for 4:3
    img.save(tmp_path / "b.jpg")
    assert mv.exif_fov_x(str(tmp_path / "b.jpg")) is None
    tall = Image.new("RGB", (300, 400))
    tall.save(tmp_path / "c.jpg", exif=exif)
    assert mv.exif_fov_x(str(tmp_path / "c.jpg")) < fov  # portrait: the short side is horizontal


def test_candidates_are_judged_against_their_anchor_view(monkeypatch, tmp_path):
    """A back view is chosen by how well it continues the 135° view it was drawn from, not by the
    photo (they share nothing); views without an anchor are still judged against the photo."""
    from PIL import Image

    paths = {}
    for name in ("photo", "a", "b0", "b1", "c0", "c1"):
        paths[name] = str(tmp_path / f"{name}.png")
        Image.new("RGB", (64, 48), (len(name) * 20, 0, 0)).save(paths[name])
    items = [{"path": paths["photo"]}, {"path": paths["a"], "label": "135° right"},
             {"path": paths["b0"], "label": "back", "candidates": [paths["b0"], paths["b1"]], "anchor": 1,
              "anchor_deg": 45, "target_deg": 180},
             {"path": paths["c0"], "label": "45° left", "candidates": [paths["c0"], paths["c1"]], "target_deg": 45}]
    imgs = [mv.load_rgb(it["path"]) for it in items]
    vin = [mv.vggt_input(im) for im in imgs]
    seen = []

    def fake_score(model, ref_vin, cand_vin, dtype, torch, target_deg=None):
        ref = next(k for k, v in enumerate(vin) if v[0] is ref_vin[0])
        seen.append((ref, target_deg))
        return float(cand_vin[0].mean()), 0.0, 45.0, 0.5

    monkeypatch.setattr(mv, "rigidity_score", fake_score)
    rep = mv.select_candidates(items, imgs, vin, None, None, None, 64)
    assert seen[:2] == [(1, 45), (1, 45)]               # back: against the 135° view, 45° away from it
    assert seen[2:] == [(0, 45), (0, 45)]               # no anchor: against the photo, 45° from it
    assert rep[0]["judged_against"] == "135° right" and rep[1]["judged_against"] == "photo"
    assert rep[0]["path"] == items[2]["path"]


def test_angle_between_ring_views():
    from twod2vr180.backends.generative import TRI_RINGS, angle_between

    assert abs(angle_between((135, 0), (180, 0)) - 45) < 1e-9
    assert abs(angle_between((270, 0), (315, 0)) - 45) < 1e-9
    assert abs(angle_between((0, 30), (0, 0)) - 30) < 1e-9
    labels = [lb for ring in TRI_RINGS for lb, *_ in ring]
    for k, ring in enumerate(TRI_RINGS):              # references always come from an earlier ring
        earlier = {lb for r in TRI_RINGS[:k] for lb, *_ in r}
        assert all(set(refs) <= earlier for *_, refs in ring)
    assert "back" in labels and len(labels) == len(set(labels)) == 8


def test_align_scale_is_robust():
    rng = np.random.default_rng(0)
    z = rng.uniform(1, 5, (50, 50))
    ref = 2.5 * z
    ref[:5] = 100       # outliers
    assert abs(mv.align_scale(z, ref, np.ones_like(z, bool)) - 2.5) < 1e-6
    assert mv.resize_map(np.arange(4.0).reshape(2, 2), 4, 4)[3, 3] == 3.0


def _scene_depth(h=120, w=160):
    """True depth (m) of a room corner with a person-sized bump 2 m away."""
    ys, xs = np.mgrid[0:h, 0:w]
    z = 3.0 + 0.4 * (xs / w) + 0.2 * (ys / h)
    bump = (xs - w / 2) ** 2 / 30 ** 2 + (ys - h / 2) ** 2 / 45 ** 2 < 1
    z[bump] = 2.0 - 0.15 * np.cos((xs[bump] - w / 2) / 30)
    return z


def test_scale_and_shift_fit_removes_the_offset_a_scale_alone_leaves():
    """A monocular depth that is right in shape but offset (z' = 0.9 z + 0.15) and partly wrong: the
    robust scale+shift fit lands within millimetres; the old scale-only fit stays centimetres off."""
    truth = _scene_depth()
    mono = (truth - 0.15) / 0.9
    rng = np.random.default_rng(0)
    bad = rng.random(truth.shape) < 0.15
    joint = np.where(bad, truth * rng.uniform(0.5, 1.5, truth.shape), truth)    # 15 % outliers
    ok = np.ones_like(truth, bool)
    s, t = mv.align_affine(mono, joint, ok)
    err_affine = np.abs(mono * s + t - truth)
    err_scale = np.abs(mono * mv.align_scale(mono, joint, ok) - truth)
    assert np.percentile(err_affine, 90) * 100 < 0.3           # cm
    assert np.percentile(err_scale, 90) * 100 > 1.5            # the subject and the walls a few cm off


def test_generated_view_depth_that_contradicts_the_joint_geometry_is_left_unknown():
    truth = _scene_depth()
    mono = truth.copy()
    mono[40:80, 20:50] *= 0.7                 # the AI view "sees" a different shape here
    valid = inside = np.ones_like(truth, bool)
    p_gen = mv.fit_prior(mono, valid, truth, inside, inside, True)
    p_photo = mv.fit_prior(mono, valid, truth, inside, inside, False)
    assert (p_gen[40:80, 20:50] == 0).mean() > 0.9 and (p_gen[:30] > 0).all()
    assert (p_photo[40:80, 20:50] > 0).all()                   # the photo keeps its own shape


def test_exif_focal_length_sets_the_photo_camera_unless_it_disagrees():
    K = np.array([[[500.0, 0, 300], [0, 500.0, 200], [0, 0, 1]]] * 2)
    out = mv.exif_intrinsics(K, {0: 60.0}, [(400, 640), (400, 640)])
    assert abs(out[0][0, 0] - 320 / np.tan(np.radians(30))) < 1e-6 and out[0][0, 2] == 320
    assert np.array_equal(out[1], K[1])                        # a generated view keeps its estimate
    far = mv.exif_intrinsics(K, {0: 120.0}, [(400, 640)] * 2)
    assert np.array_equal(far[0], K[0])                        # 2x off: the EXIF value is not trusted


def test_cropped_photos_do_not_use_the_exif_focal_length(tmp_path):
    from PIL import Image

    exif = Image.Exif()
    ifd = exif.get_ifd(0x8769)
    ifd[0xA405] = 26
    ifd[0xA002], ifd[0xA003] = 4000, 3000                    # shot at 4:3 ...
    Image.new("RGB", (400, 400)).save(tmp_path / "crop.jpg", exif=exif)      # ... saved square: cropped
    assert mv.exif_fov_x(str(tmp_path / "crop.jpg")) is None
    Image.new("RGB", (400, 300)).save(tmp_path / "resized.jpg", exif=exif)   # only resized: still valid
    assert mv.exif_fov_x(str(tmp_path / "resized.jpg")) is not None
