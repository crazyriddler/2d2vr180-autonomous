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
