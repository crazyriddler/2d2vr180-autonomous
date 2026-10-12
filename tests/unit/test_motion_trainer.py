"""Smooth 4D motion (workers/motion_trainer.py) on the CPU with the reference renderer of
test_splat_trainer: a patch moving sideways, seeded only from every second frame, must learn to be in
the right place at the frames in between."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_trainer as mt  # noqa: E402
from test_splat_trainer import ref_render  # noqa: E402

S, F = 20, 6
K = np.array([[24.0, 0, S / 2], [0, 24.0, S / 2], [0, 0, 1]])


def patch(x0, n=4, size=0.12):
    xs, ys = np.meshgrid(np.linspace(-size, size, n), np.linspace(-size, size, n))
    m = np.stack([xs.ravel() + x0, ys.ravel(), np.full(n * n, 2.0)], 1).astype(np.float32)
    k = len(m)
    return {"means": m, "scales": np.full((k, 3), 0.05, np.float32), "quats": np.tile(np.float32([1, 0, 0, 0]), (k, 1)),
            "opacity": np.full(k, 0.95, np.float32), "colors": np.tile(np.float32([1.0, 0.2, 0.2]), (k, 1))}


def x_of(k):
    return -0.4 + 0.16 * k          # the patch moves 0.16 m per frame


def frames():
    """Ground-truth images, depths and masks rendered with the reference renderer."""
    imgs, deps, masks = [], [], []
    for k in range(F):
        p = mt.to_params({**patch(x_of(k)), "vel": np.zeros((16, 3), np.float32),
                          "t_center": np.zeros(16, np.float32), "t_scale": np.ones(16, np.float32)}, "cpu")[0]
        d = {"means": p["means"], "quats": p["quats"], "scales": p["scales"], "opacities": p["opacities"],
             "sh0": p["sh0"], "shN": torch.zeros(16, 0, 3)}
        with torch.no_grad():
            out, alpha, _ = ref_render(d, torch.eye(4)[None], torch.as_tensor(K, dtype=torch.float32)[None], S, S, 0,
                                       mode="RGB+ED")
        imgs.append((out[..., :3].clamp(0, 1) * 255).round().byte().numpy())
        deps.append(np.where(alpha[..., 0].numpy() > 0.5, out[..., 3].numpy() / np.maximum(alpha[..., 0].numpy(), 1e-6), 0))
        masks.append(alpha[..., 0].numpy() > 0.05)
    return imgs, deps, masks


def error_at(params, tc, k, imgs):
    times = [(j + 0.5) / F for j in range(F)]
    m, q, s, o, sh = mt.state_at(params, tc, times[k])
    d = {"means": m, "quats": q, "scales": s, "opacities": o, "sh0": sh, "shN": torch.zeros(len(m), 0, 3)}
    with torch.no_grad():
        out, _, _ = ref_render(d, torch.eye(4)[None], torch.as_tensor(K, dtype=torch.float32)[None], S, S, 0)
    return float((out.clamp(0, 1) - torch.from_numpy(imgs[k]).float() / 255).abs().mean())


def test_seeding_gives_keyframe_lifetimes_velocities_and_a_neighbour_graph():
    times = [(j + 0.5) / F for j in range(F)]
    kfs = [{**patch(x_of(k)), "t": times[k]} for k in range(0, F, 2)]
    seed, graph = mt.seed_motion(kfs, mt.MotionConfig(knn=4))
    assert len(seed["means"]) == 48 and graph.shape == (48, 4) and graph.max() < 48
    assert np.all(graph[:16] < 16) and np.all((graph[16:32] >= 16) & (graph[16:32] < 32))   # within a keyframe
    assert np.allclose(seed["t_scale"], 0.6 * (2 / F))
    v = seed["vel"]                                                  # m per unit of scene time; truth 0.96
    assert 0.6 < float(np.median(v[:, 0])) < 1.1 and np.abs(v[:, 1:]).max() < 0.05


def test_follow_moves_each_group_as_a_whole():
    a = np.concatenate([patch(0.0)["means"], patch(1.5)["means"]])
    b = np.concatenate([patch(0.1)["means"], patch(1.5)["means"] + [0, 0.1, 0]])
    d = mt.follow(a, b)
    assert np.allclose(d[:16], [0.1, 0, 0], atol=0.02) and np.allclose(d[16:], [0, 0.1, 0], atol=0.02)


def test_motion_training_fills_the_frames_between_keyframes():
    imgs, deps, masks = frames()
    times = [(j + 0.5) / F for j in range(F)]
    kfs = [{**patch(x_of(k)), "t": times[k]} for k in range(0, F, 2)]
    seed, graph = mt.seed_motion(kfs, mt.MotionConfig(knn=4))
    params, tc = mt.to_params(seed, "cpu")
    before = np.mean([error_at(params, tc, k, imgs) for k in (1, 3, 5)])
    v0 = float(np.median(seed["vel"][:, 0]))
    cfg = mt.MotionConfig(steps=200, knn=4, rigid_samples=64, log_every=20)
    hist = mt.train_motion(params, tc, graph, imgs, [np.eye(4)] * F, [K] * F, times, masks, deps, cfg,
                           render_fn=ref_render, device="cpu")
    after = np.mean([error_at(params, tc, k, imgs) for k in (1, 3, 5)])
    assert len(hist) >= 10
    assert after < 0.7 * before, (before, after)                     # the frames between keyframes
    out = mt.export_motion(params, tc)
    assert out["motion"].shape == (len(out["means"]), 9) and out["omega"].shape == (len(out["means"]), 4)
    assert abs(float(np.median(out["motion"][:, 0])) - 0.96) < max(0.1, abs(v0 - 0.96))   # the true speed
