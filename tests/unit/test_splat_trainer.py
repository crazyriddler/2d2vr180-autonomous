"""The 3DGS optimisation loop (workers/splat_trainer.py) with a pure-PyTorch
reference renderer on the CPU (gsplat needs CUDA). Skipped without torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import splat_trainer as st  # noqa: E402


def ref_render(params, viewmat, K, width, height, sh_degree, mode="RGB"):
    """Isotropic splats, depth-sorted alpha compositing (differentiable, O(N*H*W))."""
    means = params["means"]
    R, t = viewmat[0, :3, :3], viewmat[0, :3, 3]
    pc = means @ R.T + t
    z = pc[:, 2].clamp_min(1e-4)
    k = K[0]
    u = k[0, 0] * pc[:, 0] / z + k[0, 2]
    v = k[1, 1] * pc[:, 1] / z + k[1, 2]
    sig = params["scales"].exp().mean(1) * k[0, 0] / z
    op = params["opacities"].sigmoid() * (pc[:, 2] > 1e-3)
    col = (params["sh0"][:, 0] * st.SH_C0 + 0.5).clamp(0, 1)
    order = torch.argsort(z)
    ys, xs = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    T = torch.ones(height, width)
    img = torch.zeros(height, width, 3)
    dep = torch.zeros(height, width)
    for i in order:
        g = torch.exp(-((xs - u[i]) ** 2 + (ys - v[i]) ** 2) / (2 * sig[i] ** 2 + 1e-6))
        a = (op[i] * g).clamp(max=0.99)
        img = img + (T * a)[..., None] * col[i]
        dep = dep + T * a * z[i]
        T = T * (1 - a)
    alpha = (1 - T)[..., None]
    if mode == "ED":
        out = dep[..., None]
    elif mode == "RGB+ED":
        out = torch.cat([img, dep[..., None]], -1)
    else:
        out = img
    return out, alpha, {}


def look_at(eye, target):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, -1.0, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, eye
    return m


def make_views(n=4, size=24):
    K = np.array([[30.0, 0, size / 2], [0, 30.0, size / 2], [0, 0, 1]])
    c2ws = [look_at(np.array([0.4 * np.sin(a), 0, 0.4 * (1 - np.cos(a))]), [0, 0, 2])
            for a in np.linspace(-0.3, 0.3, n)]
    return np.stack(c2ws), np.stack([K] * n), [(size, size)] * n


def gt_params():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.uniform([-0.6, -0.6, 2.0], [0.6, 0.6, 2.0], (40, 3)),
                          rng.uniform([-0.3, -0.3, 1.5], [0.3, 0.3, 1.5], (12, 3))]).astype(np.float32)
    cols = rng.uniform(0, 1, (len(pts), 3)).astype(np.float32)
    cfg = st.TrainConfig(sh_degree=0, init_points=1000)
    p = st.init_params(pts, cols, cfg, "cpu")
    with torch.no_grad():
        p["opacities"].fill_(3.0)
    return p, pts, cols


def test_training_reduces_loss_and_fits_views():
    gt, pts, cols = gt_params()
    c2ws, Ks, sizes = make_views()
    vm = torch.linalg.inv(torch.as_tensor(c2ws, dtype=torch.float32))
    Kt = torch.as_tensor(Ks, dtype=torch.float32)
    with torch.no_grad():
        imgs = [(ref_render(gt, vm[i:i + 1], Kt[i:i + 1], 24, 24, 0)[0].clamp(0, 1) * 255).round().byte().numpy()
                for i in range(len(c2ws))]
    rng = np.random.default_rng(1)
    noisy = pts + rng.normal(0, 0.03, pts.shape).astype(np.float32)
    cfg = st.TrainConfig(steps=150, sh_degree=0, densify=False, coarse_until=0.0, ssim_weight=0.2,
                         lr_sh0=0.05, log_every=10)
    params, hist = st.train(imgs, c2ws, Ks, [3, 1, 1, 1], noisy, np.full_like(cols, 0.5), cfg, device="cpu",
                            render_fn=ref_render)
    first = np.mean([h[1] for h in hist[:2]])
    last = np.mean([h[1] for h in hist[-2:]])
    assert last < 0.6 * first
    with torch.no_grad():
        out = ref_render(params, vm[:1], Kt[:1], 24, 24, 0)[0].clamp(0, 1)
    assert st.psnr(out, torch.from_numpy(imgs[0]).float() / 255) > 18


def test_provenance_from_real_view_visibility():
    params, pts, _ = gt_params()
    c2ws, Ks, sizes = make_views(3)
    vm = torch.linalg.inv(torch.as_tensor(c2ws, dtype=torch.float32))
    Kt = torch.as_tensor(Ks, dtype=torch.float32)

    def depth(v):
        with torch.no_grad():
            d, a, _ = ref_render(params, vm[v:v + 1], Kt[v:v + 1], 24, 24, 0, mode="ED")
        d = d[..., 0] / a[..., 0].clamp_min(1e-6)
        return torch.where(a[..., 0] > 0.5, d, torch.full_like(d, float("inf")))

    # a splat far outside every view frustum is never seen; the small front splats are moved
    # out of the way so the plane is unoccluded, except one directly behind a big occluder
    with torch.no_grad():
        params["means"][0] = torch.tensor([50.0, 0.0, 2.0])
        params["means"][40:] = torch.tensor([0.0, 0.0, -5.0])
        params["means"][1] = torch.tensor([0.0, 0.0, 6.0])        # hidden behind the plane
    counts = st.visibility_counts(params, c2ws, Ks, sizes, [0, 1], depth, tol=0.1)
    assert counts[0] == 0
    assert (counts[2:40] == 2).float().mean() > 0.8          # plane splats are seen by both real views
    assert counts[1] == 0
    codes = st.provenance_codes(counts, any_generated=True)
    assert codes[0] == 2 and (codes == 0).any()
    assert set(st.provenance_codes(counts, any_generated=False).tolist()) <= {0, 1}


def test_ply_round_trip_with_provenance(tmp_path):
    from twod2vr180.scene import read_gaussian_ply

    params, pts, cols = gt_params()
    codes = torch.zeros(len(pts), dtype=torch.uint8)
    codes[:5] = 2
    n = st.write_ply(params, str(tmp_path / "s.ply"), codes)
    sc = read_gaussian_ply(tmp_path / "s.ply")
    assert n == len(sc) == len(pts)
    assert np.allclose(sc.means, pts, atol=1e-6)
    assert np.allclose(sc.colors, np.clip(cols, 0, 1), atol=1e-4)
    assert (sc.provenance[:5] == 2).all() and (sc.provenance[5:] == 0).all()


def test_ssim_identity_and_knn():
    a = torch.rand(1, 3, 32, 32)
    assert abs(float(st.ssim(a, a)) - 1) < 1e-5
    assert float(st.ssim(a, torch.rand(1, 3, 32, 32))) < 0.5
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    d = st.knn_scales(pts, k=2)
    assert torch.allclose(d, torch.tensor([1.0, 1.0, 2.0, 3.0]))


def _scene():
    gt, pts, cols = gt_params()
    c2ws, Ks, sizes = make_views()
    vm = torch.linalg.inv(torch.as_tensor(c2ws, dtype=torch.float32))
    Kt = torch.as_tensor(Ks, dtype=torch.float32)
    with torch.no_grad():
        imgs = [(ref_render(gt, vm[i:i + 1], Kt[i:i + 1], 24, 24, 0)[0].clamp(0, 1) * 255).round().byte().numpy()
                for i in range(len(c2ws))]
        deps = [ref_render(gt, vm[i:i + 1], Kt[i:i + 1], 24, 24, 0, mode="ED")[0][..., 0].numpy()
                for i in range(len(c2ws))]
    return gt, pts, cols, c2ws, Ks, vm, Kt, imgs, deps


def test_low_confidence_pixels_do_not_corrupt_the_model():
    gt, pts, cols, c2ws, Ks, vm, Kt, imgs, _ = _scene()
    bad = [im.copy() for im in imgs]
    for i in (1, 2, 3):                       # "generated" views with a changed region (e.g. a turned head)
        bad[i][6:18, 6:18] = 255
    conf = [None] + [np.ones((24, 24), np.float32) for _ in range(3)]
    for c in conf[1:]:
        c[6:18, 6:18] = 0.0
    cfg = st.TrainConfig(steps=150, sh_degree=0, densify=False, coarse_until=0.0, lr_sh0=0.05, log_every=50)
    rng = np.random.default_rng(1)
    noisy = pts + rng.normal(0, 0.03, pts.shape).astype(np.float32)

    def fit(pixel_weights):
        p, _ = st.train(bad, c2ws, Ks, [1, 1, 1, 1], noisy, np.full_like(cols, 0.5), cfg, device="cpu",
                        render_fn=ref_render, pixel_weights=pixel_weights)
        with torch.no_grad():
            out = ref_render(p, vm[1:2], Kt[1:2], 24, 24, 0)[0].clamp(0, 1)
        return st.psnr(out, torch.from_numpy(imgs[1]).float() / 255)    # against the CLEAN view

    assert fit(conf) > fit(None) + 1.0


def test_depth_prior_runs_and_keeps_geometry():
    gt, pts, cols, c2ws, Ks, vm, Kt, imgs, deps = _scene()
    cfg = st.TrainConfig(steps=60, sh_degree=0, densify=False, coarse_until=0.0, lr_sh0=0.05, log_every=20,
                         depth_weight=0.5, depth_until=1.0)
    rng = np.random.default_rng(2)
    noisy = pts + rng.normal(0, 0.05, pts.shape).astype(np.float32)
    p, hist = st.train(imgs, c2ws, Ks, [1, 1, 1, 1], noisy, np.full_like(cols, 0.5), cfg, device="cpu",
                       render_fn=ref_render, depth_priors=deps)
    with torch.no_grad():
        d, a, _ = ref_render(p, vm[:1], Kt[:1], 24, 24, 0, mode="ED")
    m = (a[..., 0] > 0.5).numpy() & (deps[0] > 0)
    assert np.isfinite(hist[-1][1])
    assert np.median(np.abs(d[..., 0].numpy()[m] - deps[0][m]) / deps[0][m]) < 0.05


def test_cleanup_removes_junk_but_keeps_the_scene():
    gt, pts, cols = gt_params()
    c2ws, Ks, sizes = make_views()
    extra = np.array([[0, 0, -5.0],          # behind every camera: never seen
                      [0.2, 0.1, 1.8],        # a floater far from its neighbours? (kept if not isolated)
                      [30.0, 30.0, 40.0]], np.float32)   # far away and alone
    p = st.init_params(np.concatenate([pts, extra]), np.concatenate([cols, np.full((3, 3), 0.5, np.float32)]),
                       st.TrainConfig(sh_degree=0, init_points=1000), "cpu")
    with torch.no_grad():
        p["opacities"].fill_(3.0)
        p["opacities"][len(pts) - 1] = -8.0                       # nearly transparent
        p["scales"][len(pts) - 2] = p["scales"][len(pts) - 2] + 6  # a huge blob
    keep = st.cleanup_mask(p, c2ws, Ks, sizes)
    assert not keep[len(pts)] and not keep[len(pts) + 2]          # unseen, isolated
    assert not keep[len(pts) - 1] and not keep[len(pts) - 2]       # transparent, oversized
    assert keep[: len(pts) - 2].float().mean() > 0.9               # the scene stays
    sub = st.subset(p, keep)
    assert len(sub["means"]) == int(keep.sum()) and set(sub.keys()) == set(p.keys())


def test_colour_drift_of_generated_views_is_absorbed_by_appearance():
    gt, pts, cols, c2ws, Ks, vm, Kt, imgs, _ = _scene()
    drift = [im.copy() for im in imgs]
    for i in (1, 2, 3):                       # "generated" views: darker and warmer than the photo
        f = drift[i].astype(np.float32) / 255
        f = f * np.array([0.95, 0.75, 0.6]) + np.array([0.08, 0.02, 0.0])
        drift[i] = (f.clip(0, 1) * 255).round().astype(np.uint8)
    cfg = st.TrainConfig(steps=200, sh_degree=0, densify=False, coarse_until=0.0, lr_sh0=0.05, log_every=50,
                         lr_appearance=5e-3)
    rng = np.random.default_rng(1)
    noisy = pts + rng.normal(0, 0.03, pts.shape).astype(np.float32)

    def fit(appearance):
        out = []
        p, _ = st.train(drift, c2ws, Ks, [1, 1, 1, 1], noisy, np.full_like(cols, 0.5), cfg, device="cpu",
                        render_fn=ref_render, appearance=appearance, appearance_out=out)
        with torch.no_grad():
            r = ref_render(p, vm[2:3], Kt[2:3], 24, 24, 0)[0].clamp(0, 1)
        return st.psnr(r, torch.from_numpy(imgs[2]).float() / 255), out   # against the photo's colours

    base, _ = fit(None)
    corrected, out = fit([False, True, True, True])
    assert corrected > base + 1.0
    assert out[0] is None and out[1] is not None
    assert out[1][0][2][2] < 0.95             # it learned to darken blue for the generated views
