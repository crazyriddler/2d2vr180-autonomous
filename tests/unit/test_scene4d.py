"""4D scenes (scene4d.py): time-dependent splats, .splatv round trip, saving / loading, CLI viewer check."""

import json
from pathlib import Path

import numpy as np

from twod2vr180.scene import Camera, GaussianScene
from twod2vr180.scene4d import (DynamicSplats, Scene4D, export_dir_of, frame_time, is_scene4d, merge,
                                per_frame_splats, read_splatv, splatv_for, write_splatv)


def _scene(n, seed=0, z=3.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    m = rng.normal(size=(n, 3)).astype(np.float32)
    m[:, 2] += z
    return GaussianScene(m, rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32), q,
                         rng.uniform(0.2, 1, n).astype(np.float32), rng.uniform(0, 1, (n, 3)).astype(np.float32),
                         cameras=[Camera(64, 48, 50, 50, 32, 24)], metric_scale=True)


def _s4(frames=8):
    dy = DynamicSplats.concat([per_frame_splats(_scene(30, k + 1, 1.5), k, frames) for k in range(frames)])
    cams = [Camera(64, 48, 50, 50, 32, 24, np.eye(4)) for _ in range(frames)]
    for k, c in enumerate(cams):
        c.c2w[0, 3] = 0.1 * k
    return Scene4D(_scene(500), dy, 10.0, frames, cams, {"playback": "frames", "camera_motion": "moving"})


def test_per_frame_splats_are_visible_only_at_their_frame():
    s4 = _s4()
    for k in range(s4.frames):
        vis = s4.dynamic.at(s4.time_of(k))
        assert len(vis) == 30 and np.allclose(vis.opacity, s4.dynamic.opacity[30 * k:30 * k + 30], rtol=1e-5)
        assert len(s4.frame(k)) == 530
    assert len(s4.dynamic.at(frame_time(2, 8) + 0.5 / 8)) == 60      # half-way: both neighbours, faded
    assert s4.duration_s == 0.8


def test_motion_and_rotation_follow_the_polynomial():
    base = per_frame_splats(_scene(5), 0, 1)
    base.t_center[:] = 0.5
    base.t_scale[:] = 100.0
    base.motion = np.zeros((5, 9), np.float32)
    base.motion[:, 0] = 2.0                 # 2 units per unit of time along x
    base.motion[:, 4] = 1.0                 # + t^2 along y
    a, b = base.at(0.5), base.at(0.75)
    assert np.allclose(b.means[:, 0] - a.means[:, 0], 0.5, atol=1e-6)
    assert np.allclose(b.means[:, 1] - a.means[:, 1], 0.0625, atol=1e-6)


def test_splatv_round_trip_and_layout(tmp_path):
    s4 = _s4()
    p = write_splatv(s4, tmp_path / "scene.splatv")
    raw = p.read_bytes()
    magic, jlen = np.frombuffer(raw[:8], "<u4")
    meta = json.loads(raw[8:8 + jlen])[0]
    assert magic == 0x674B and meta["type"] == "splat" and meta["texwidth"] == 4096
    assert meta["size"] == 4096 * meta["texheight"] * 4 * 4 and len(raw) == 8 + jlen + meta["size"]
    assert meta["count"] == 740 and meta["background_count"] == 500 and meta["frames"] == 8
    assert meta["cameras"][0]["rotation"] == np.eye(3).tolist()
    r = read_splatv(p)
    assert len(r.background) == 500 and len(r.dynamic) == 240 and r.frames == 8 and r.fps == 10.0
    assert np.array_equal(r.background.means, s4.background.means)          # positions are float32
    assert np.allclose(r.background.scales, s4.background.scales, rtol=2e-3)  # the rest is half precision
    assert np.allclose(r.background.colors, s4.background.colors, atol=1 / 255)
    assert np.allclose(np.abs((r.background.quats * s4.background.quats).sum(1)), 1, atol=2e-3)
    assert np.allclose(r.dynamic.t_center, s4.dynamic.t_center, atol=5e-4)
    for k in range(8):
        assert len(r.dynamic.at(r.time_of(k))) == 30
    assert np.allclose([c.c2w[0, 3] for c in r.cameras], [0.1 * k for k in range(8)])


def test_splatv_budget_keeps_the_most_important_splats(tmp_path):
    s4 = _s4()
    r = read_splatv(write_splatv(s4, tmp_path / "s.splatv", budget=300))
    assert len(r.background) + len(r.dynamic) == 300 and len(r.background) >= 120
    imp = np.prod(s4.background.scales, 1) * s4.background.opacity
    keep = np.sort(np.argsort(-imp)[:len(r.background)])          # the largest, most opaque splats
    assert np.array_equal(r.background.means, s4.background.means[keep])


def test_save_load_and_helpers(tmp_path):
    s4 = _s4()
    out = s4.save(tmp_path / "export")
    exp = tmp_path / "export"
    assert Path(out["scene_ply"]) == exp / "scene.ply" and Path(out["splatv"]).exists()
    for p in (exp, exp / "scene.ply", Path(out["scene4d"]), exp / "scene.splatv"):
        assert is_scene4d(p)
        assert export_dir_of(p) == exp
        l4 = Scene4D.load(p)
        assert l4.frames == 8 and len(l4.dynamic) == 240 and len(l4.background) == 500
    assert not is_scene4d(tmp_path / "other.ply")
    (exp / "scene.splatv").unlink()
    assert splatv_for(exp / "scene.ply").exists()                 # rewritten when missing
    m = merge(s4.background, s4.dynamic.at(s4.time_of(0)))
    assert len(m) == 530 and m.metric_scale and m.cameras


def test_cli_serves_a_4d_scene_to_the_4d_player(tmp_path, capsys):
    from twod2vr180 import cli

    _s4().save(tmp_path / "export")
    rc = cli.main(["view-vr", str(tmp_path / "export" / "scene.ply"), "--check"])
    out = capsys.readouterr().out
    assert rc == 0 and "player4d.html" in out and "VR viewer OK" in out
