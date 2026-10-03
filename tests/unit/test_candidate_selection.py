import math
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_hopeless_generated_views_are_dropped_but_one_is_kept():
    items = [{"path": "photo"}, {"generated": True, "label": "a", "rigidity": 0.3},
             {"generated": True, "label": "b", "rigidity": 0.95}, {"generated": True, "label": "c", "rigidity": 0.9}]
    imgs, vin = list("PABC"), list("pabc")
    dropped = mv.drop_inconsistent(items, imgs, vin, 0.8)
    assert sorted(dropped) == ["b", "c"] and [it.get("label") for it in items] == [None, "a"] and imgs == ["P", "A"]
    items = [{"path": "photo"}, {"generated": True, "label": "a", "rigidity": 0.9},
             {"generated": True, "label": "b", "rigidity": 0.85}]
    imgs, vin = list("PAB"), list("pab")
    mv.drop_inconsistent(items, imgs, vin, 0.8)
    assert [it.get("label") for it in items] == [None, "b"]          # the least bad one stays


def test_da3_stub_serves_any_helper_the_model_files_import(monkeypatch):
    for name in ("depth_anything_3.utils.export", "depth_anything_3.utils.pose_align"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    mv.da3_stub()
    # the Giant / Nested models' Gaussian head imports this one (rc29 failed on it)
    from depth_anything_3.utils.pose_align import align_poses_umeyama, batch_align_poses_umeyama  # noqa: F401
    from depth_anything_3.utils.export import export  # noqa: F401
    mod = sys.modules["depth_anything_3.utils.pose_align"]
    assert not hasattr(mod, "__file__") and not hasattr(mod, "__path__")
    for name in ("depth_anything_3.utils.export", "depth_anything_3.utils.pose_align"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_da3_that_cannot_load_falls_back_to_vggt(monkeypatch):
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    def broken(*a, **k):
        raise ImportError("cannot import name 'something' from 'depth_anything_3'")

    logs = []
    monkeypatch.setattr(mv, "load_da3", broken)
    monkeypatch.setattr(mv, "log", logs.append)
    assert mv.try_load_da3("x", "cuda", FakeTorch) is None
    assert "using VGGT" in logs[0] and "ImportError" in logs[0]


def _person_in_front_of_wall(h=300, w=200):
    d = np.full((h, w), 3.0)                       # backdrop wall at 3 m, slightly tilted
    d += np.linspace(0, 0.2, h)[:, None]
    yy, xx = np.mgrid[0:h, 0:w]
    person = ((xx - 80) / 45.0) ** 2 + ((yy - 170) / 120.0) ** 2 < 1
    d[person] = 1.6 + 0.1 * np.sin(xx[person] / 9.0)
    return d, person


def test_subject_mask_finds_the_person_in_front_of_the_wall():
    pytest.importorskip("scipy")
    d, person = _person_in_front_of_wall()
    m, why = mv.subject_mask(d)
    assert m is not None, why
    assert m[person].mean() > 0.98                  # the whole subject (slightly dilated)
    assert m[~person].mean() < 0.05                 # and hardly any wall


def test_no_subject_mode_for_a_scene_without_a_separate_subject():
    pytest.importorskip("scipy")
    h, w = 300, 200
    ground = 1.0 + 9.0 * (1 - np.linspace(0, 1, h))[:, None] * np.ones((1, w))   # a landscape / floor
    m, why = mv.subject_mask(ground)
    assert m is None and "no separate subject" in why
    assert mv.subject_mask(np.full((h, w), 2.0))[0] is None
    room = np.exp(np.random.default_rng(0).uniform(0, 2, (h, w)))          # depths spread evenly
    assert mv.subject_mask(room)[0] is None
    d, person = _person_in_front_of_wall()
    d[220:] = np.linspace(3.0, 1.2, 80)[:, None]                            # standing on a floor
    assert mv.subject_mask(d)[0] is not None


def test_masked_flattens_the_background():
    img = np.full((40, 30, 3), 200, np.uint8)
    m = np.zeros((40, 30), bool)
    m[10:30, 5:20] = True
    out = mv.masked(img, m)
    assert (out[m] == 200).all() and (out[~m] == 128).all()


def test_turntable_orbits_the_subject_and_stays_within_the_views():
    d, person = _person_in_front_of_wall()
    h, w = d.shape
    K = np.array([[200.0, 0, w / 2], [0, 200.0, h / 2], [0, 0, 1]])
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([(xs + .5 - w / 2) / 200 * d, (ys + .5 - h / 2) / 200 * d, d], -1).reshape(-1, 3)
    c2w0 = np.eye(4)

    def yawed(deg):
        a = math.radians(deg)
        m = np.eye(4)
        m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        return m

    zc, yaw = mv.turntable_orbit(pts, [c2w0, yawed(40), yawed(-12)], K, (w, h), person)
    assert 1.5 < zc < 1.75                          # around the person, not the wall at 3 m
    assert abs(yaw - 28) < 0.5                      # 70 % of the widest view (40°)
    zc_all, yaw_small = mv.turntable_orbit(pts, [c2w0, yawed(7)], K, (w, h))
    assert zc_all > 2.5 and yaw_small == 8.0        # no mask: median of everything; small views: small swing


def test_subject_mode_leaves_out_generated_candidates_without_a_subject(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(mv, "log", lambda *a: None)
    good = tmp_path / "good.png"
    Image.fromarray(np.full((8, 6, 3), 90, np.uint8)).save(good)
    m = np.ones((8, 6), bool)
    items = [{"path": "photo.png"},
             {"path": "a0.png", "generated": True, "candidates": ["a0.png", str(good)]},   # first has no subject
             {"path": "b0.png", "generated": True, "candidates": ["b0.png", "b1.png"]}]    # none has one
    imgs = ["photo", "a0", "b0"]
    masks = {"photo.png": m, "a0.png": None, str(good): m, "b0.png": None, "b1.png": None}
    assert mv.keep_masked_views(items, imgs, masks, 100) is masks
    assert [it["path"] for it in items] == ["photo.png", str(good)]
    assert items[1]["candidates"] == [str(good)] and imgs[0] == "photo" and imgs[1].shape == (8, 6, 3)
    # no generated view with a subject left: subject mode is abandoned, nothing is removed
    items2 = [{"path": "photo.png"}, {"path": "b0.png", "generated": True}]
    assert mv.keep_masked_views(items2, ["p", "b"], masks, 100) is None and len(items2) == 2


def _rot(deg, axis=1):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def test_feedforward_gaussians_are_brought_into_the_camera_frame():
    # model frame: photo camera at some pose; metric scaling happens after the Gaussians are built
    c2w_model = np.eye(4)
    c2w_model[:3, :3] = _rot(20) @ _rot(-10, 0)
    c2w_model[:3, 3] = [0.3, -0.2, 0.5]
    s = 2.5
    pts_cam = np.array([[0.1, 0.2, 1.0], [-0.3, 0.1, 2.0]])           # in the photo camera (metric)
    means_model = (pts_cam / s) @ c2w_model[:3, :3].T + c2w_model[:3, 3]     # model units (unscaled)
    c2w_metric = c2w_model.copy()
    c2w_metric[:3, 3] *= s                                           # what DA3 returns as extrinsics
    inv0 = np.linalg.inv(c2w_metric)
    q_cam = np.array([[1.0, 0, 0, 0], [0.9238795, 0.3826834, 0, 0]])   # orientations in the photo camera
    q_model = mv.quat_mul(mv.quat_from_rotmat(c2w_model[:3, :3]), q_cam)
    means, scales, quats = mv.gaussians_to_frame(means_model, np.full((2, 3), 0.01), q_model, s, inv0)
    assert np.allclose(means, pts_cam, atol=1e-5)
    assert np.allclose(scales, 0.025)
    assert np.allclose(np.abs((quats * q_cam).sum(1)), 1, atol=1e-5)  # same rotation (up to sign)


def test_feedforward_init_keeps_the_subject_and_adds_the_photo_background():
    V, h, w = 2, 32, 24
    G = {"means": np.random.default_rng(0).normal(size=(V, h, w, 3)).astype(np.float32),
         "scales": np.full((V, h, w, 3), 0.01, np.float32), "quats": np.tile([1, 0, 0, 0], (V, h, w, 1)).astype(
             np.float32), "dc": np.zeros((V, h, w, 3), np.float32), "opacity": np.full((V, h, w), 0.6, np.float32)}
    subj_hw = np.zeros((h, w), bool)
    subj_hw[8:24, 6:18] = True
    vin = [(None, {"mask": subj_hw}), (None, {"mask": subj_hw})]
    depth = [np.full((h, w), 2.0)] * 2
    photo_depth = np.full((64, 48), 3.0, np.float32)
    K0 = np.array([[50.0, 0, 24], [0, 50.0, 32], [0, 0, 1]])
    g = mv.ff_init_gaussians(G, [{}, {"generated": True}], vin, depth, [subj_hw, subj_hw], photo_depth,
                             np.full((64, 48, 3), 100, np.uint8), K0, np.eye(4))
    n_subject = 2 * int(subj_hw.sum())
    assert len(g["means"]) > n_subject                      # + the photo's background
    bg = g["means"][n_subject:]
    assert np.allclose(bg[:, 2], 3.0) and np.allclose(g["opacity"][n_subject:], 0.9)
    # without subject mode: everything but the border and the farthest 10 %
    g2 = mv.ff_init_gaussians(G, [{}, {"generated": True}], [(None, {}), (None, {})],
                              [np.linspace(1, 5, h * w).reshape(h, w)] * 2, [None, None], None, None, K0, np.eye(4))
    assert 0.6 * 2 * h * w < len(g2["means"]) < 0.9 * 2 * h * w


def test_duplicate_surfaces_of_generated_views_are_removed():
    """Each view brings its own copy of a surface (ghosts in the turntable): the photo's copy is kept,
    a generated view only adds what the photo does not show (here: what lies behind the photo's wall)."""
    h, w = 20, 20
    K = np.array([[20.0, 0, 10], [0, 20.0, 10], [0, 0, 1]])
    photo_depth = np.full((h, w), 2.0)
    valid = np.ones((h, w), bool)
    X = np.array([[0.0, 0.0, 2.03],      # the same wall, 1.5 % off: duplicate
                  [0.1, 0.0, 1.5],       # in front of the wall the photo sees: contradicts the photo
                  [0.0, 0.1, 2.6],       # behind the wall: hidden from the photo, kept
                  [5.0, 0.0, 2.0]])      # outside the photo's frame: kept
    dup = mv.surface_duplicates(X, K, np.eye(4), photo_depth, valid, 0.04, in_front_too=True)
    assert dup.tolist() == [True, True, False, False]
    assert mv.surface_duplicates(X, K, np.eye(4), photo_depth, valid, 0.04).tolist() == [True, False, False, False]


def test_feedforward_init_drops_generated_copies_and_uses_the_photo_layer():
    V, h, w = 2, 16, 16
    K = np.array([[16.0, 0, 8], [0, 16.0, 8], [0, 0, 1]])
    ys, xs = np.mgrid[0:h, 0:w]
    plane = np.stack([(xs + 0.5 - 8) / 16 * 2, (ys + 0.5 - 8) / 16 * 2, np.full((h, w), 2.0)], -1)
    means = np.stack([plane, plane * [1, 1, 1.01]]).astype(np.float32)   # view 1 sees the same wall
    G = {"means": means, "scales": np.full((V, h, w, 3), 0.01, np.float32),
         "quats": np.tile([1, 0, 0, 0], (V, h, w, 1)).astype(np.float32), "dc": np.zeros((V, h, w, 3), np.float32),
         "opacity": np.full((V, h, w), 0.6, np.float32)}
    depth = [np.full((h, w), 2.0)] * 2
    items = [{}, {"generated": True}]
    vin = [(None, {}), (None, {})]
    plain = mv.ff_init_gaussians(G, items, vin, depth, [None, None], None, None, K, np.eye(4), far_pct=100)
    dedup = mv.ff_init_gaussians(G, items, vin, depth, [None, None], None, None, K, np.eye(4), far_pct=100,
                                 cams=[(K, np.eye(4))] * 2)
    assert len(dedup["means"]) == len(plain["means"]) // 2           # the generated view's copy is gone
    # photo layer: one Gaussian per photo pixel (at 4x DA3's resolution) instead of DA3's for the photo
    Kp = K * [[4], [4], [1]]
    layer = np.full((64, 64), 2.0, np.float32)
    layer[:, :2] = 0                                                 # e.g. a depth jump left out
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    g = mv.ff_init_gaussians(G, items, vin, depth, [None, None], None, img, Kp, np.eye(4), far_pct=100,
                             cams=[(K, np.eye(4))] * 2, photo_layer=layer)
    assert len(g["means"]) == 64 * 62
    assert np.allclose(g["dc"][0] * 0.28209479177387814 + 0.5, img[0, 2] / 255, atol=1e-5)


def test_photo_layer_depth_is_smooth_and_skips_depth_jumps():
    d = np.full((10, 10), 2.0)
    d[:, 5:] = 4.0                                                  # a jump from 2 m to 4 m
    d[:5, :5] = np.linspace(2.0, 2.05, 25).reshape(5, 5)
    info = {"sx": 1.0, "sy": 1.0, "crop": 0, "w": 10}
    out = mv.photo_layer_depth(d, info, (40, 40), None)
    assert out.shape == (40, 40)
    assert (out[:, 18:22] == 0).all()                               # around the jump: left out
    assert (out[:, :12] > 0).all() and (out[:, 26:] > 0).all()
    assert len(np.unique(out[:12, :12])) > 25                       # bilinear, not 2.5-px stairs


def _look(eye, target):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, -1.0, 0])
    r /= np.linalg.norm(r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, np.cross(f, r), f, eye
    return m


def _wall_and_subject(cams, K, h, w):
    """Depth maps of a subject (a 0.3 m square at z = 1) in front of a wall (z = 2), seen by `cams`."""
    dd, masks = [], []
    ys, xs = np.mgrid[0:h, 0:w]
    for c2w in cams:
        d_cam = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
        d_w = d_cam @ c2w[:3, :3].T
        o = c2w[:3, 3]
        t_wall = (2.0 - o[2]) / d_w[..., 2]
        t_subj = (1.0 - o[2]) / d_w[..., 2]
        p = o + d_w * t_subj[..., None]
        on_subj = (np.abs(p[..., 0]) < 0.15) & (np.abs(p[..., 1]) < 0.15)
        t = np.where(on_subj, t_subj, t_wall)
        dd.append((t * d_cam[..., 2] / np.linalg.norm(d_cam, axis=-1) * np.linalg.norm(d_cam, axis=-1)).astype(np.float32))
        masks.append(on_subj)
    return dd, masks


def test_mesh_keeps_the_subject_apart_from_the_wall_behind_it(tmp_path):
    o3d = pytest.importorskip("open3d")
    h, w = 120, 160
    K = np.array([[150.0, 0, 80], [0, 150.0, 60], [0, 0, 1]])
    cams = [_look(np.array([x, 0, 0.0]), [0, 0, 1.0]) for x in (0.0, -0.4, 0.4)]
    dd, masks = _wall_and_subject(cams, K, h, w)
    cc = [np.full((h, w, 3), 128, np.uint8)] * 3
    out = mv.layered_mesh(dd, [K] * 3, cams, cc, masks, str(tmp_path))
    assert out["subject"]["voxel_m"] < 0.003 and out["subject"]["faces"] > 1000
    z = -np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "scene.obj")).vertices)[:, 2]    # OBJ is z-flipped
    assert ((z > 0.9) & (z < 1.1)).any() and ((z > 1.9) & (z < 2.1)).any()
    between = (z > 1.15) & (z < 1.85)
    assert between.mean() < 0.01, between.mean()             # no skin joining the subject to the wall
    zs = -np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "subject.obj")).vertices)[:, 2]
    assert np.all(np.abs(zs - 1.0) < 0.05)


def test_mesh_voxel_respects_the_face_budget():
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]])
    far = [np.full((800, 1000), 10.0, np.float32)]            # 80 m² of wall 10 m away
    v = mv.budget_voxel(far, [K], 0.001, 500_000)
    assert 2 * mv.surface_area(far, [K]) / v ** 2 <= 500_001
    assert mv.budget_voxel([np.full((8, 10), 1.0, np.float32)], [K], 0.001, 500_000) == 0.001   # small: as asked


def test_back_views_are_judged_by_angle_not_dropped():
    """A view of the subject's back shares nothing with the photo: it must not score as inconsistent."""
    h, w = 20, 20
    vin0 = (np.zeros((h, w, 3), np.float32), {})
    pred = {"pads": (0, 0), "depth": [np.full((h, w), 2.0), np.full((h, w), 2.0)], "conf": [np.ones((h, w))] * 2,
            "K": [np.array([[20.0, 0, 10], [0, 20.0, 10], [0, 0, 1]])] * 2,
            "w2c": [np.eye(4)[:3], np.diag([-1.0, 1.0, -1.0, 1.0])[:3] + np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                                                                  [0, 0, 0, 4.0]])]}
    sc, err, ang, ov = mv.rigidity_score(None, vin0, vin0, None, None, 180, predict=lambda a: pred)
    assert abs(ang - 180) < 1 and sc < 0.8 and err is None          # kept (drop threshold 0.8)
    sc45, *_ = mv.rigidity_score(None, vin0, vin0, None, None, 45, predict=lambda a: pred)
    assert sc45 == 10.0                                             # a 45° view with no overlap stays suspect
