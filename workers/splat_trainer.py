"""3D Gaussian Splatting optimisation from posed images (runtime 'recon3d-cu124').

Plain PyTorch around gsplat's rasteriser so the loop can be tested with a
reference renderer on the CPU. Conventions: cameras are OpenCV camera-to-world
4x4 matrices; intrinsics are in pixels of the image they belong to; scene
coordinates are the reference camera's frame (so SH coefficients never need
rotating).

Improvements over a minimal trainer:
  * per-view sampling weights (the user's real photo counts more than generated views),
  * D-SSIM + L1 photometric loss, coarse-to-fine resolution schedule,
  * scale regularisation (no needle splats that shimmer in VR) and opacity decay,
  * densification (gsplat DefaultStrategy) with a hard cap on the splat count,
  * per-splat provenance: how many *real* (non-generated) views see each splat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SH_C0 = 0.28209479177387814


@dataclass
class TrainConfig:
    steps: int = 10000
    sh_degree: int = 3
    max_gaussians: int = 2_000_000
    init_points: int = 300_000
    ssim_weight: float = 0.2
    scale_reg: float = 0.01        # penalises max/min scale ratios above `max_aniso`
    max_aniso: float = 10.0
    opacity_reg: float = 0.0005
    densify: bool = True
    coarse_until: float = 0.3      # fraction of steps trained at half resolution
    lr_means: float = 1.6e-4       # times scene scale
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    seed: int = 0
    log_every: int = 250


def knn_scales(points, k: int = 4):
    """Mean distance to the k-1 nearest neighbours (chunked brute force, torch)."""
    import torch

    n = len(points)
    out = torch.empty(n, device=points.device)
    chunk = max(1, int(2e8 // max(n, 1)))
    for s in range(0, n, chunk):
        d = torch.cdist(points[s:s + chunk], points)
        out[s:s + chunk] = d.topk(k, largest=False).values[:, 1:].mean(1)
    return out


def knn_scales_fast(points, k: int = 4):
    """KNN distances: scikit-learn KD-tree when available (runtime has it), else brute force."""
    import torch

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return knn_scales(points, k)
    x = points.detach().cpu().numpy()
    d, _ = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
    return torch.from_numpy(d[:, 1:].mean(1).astype(np.float32)).to(points.device)


def init_params(points: np.ndarray, colors: np.ndarray, cfg: TrainConfig, device):
    import torch

    g = torch.Generator().manual_seed(cfg.seed)
    pts = torch.as_tensor(points, dtype=torch.float32)
    cols = torch.as_tensor(colors, dtype=torch.float32).clamp(0, 1)
    if len(pts) > cfg.init_points:
        idx = torch.randperm(len(pts), generator=g)[:cfg.init_points]
        pts, cols = pts[idx], cols[idx]
    pts, cols = pts.to(device), cols.to(device)
    dist = knn_scales_fast(pts).clamp_min(1e-7)
    n = len(pts)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1
    sh = torch.zeros(n, (cfg.sh_degree + 1) ** 2, 3, device=device)
    sh[:, 0] = (cols - 0.5) / SH_C0
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=device))),
        "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
    })


def ssim(a, b):
    """Mean SSIM of two (1,3,H,W) images in [0,1] (11x11 Gaussian window)."""
    import torch
    import torch.nn.functional as F

    x = torch.arange(11, dtype=a.dtype, device=a.device) - 5
    w1 = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
    w1 = w1 / w1.sum()
    win = (w1[:, None] * w1[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda t: F.conv2d(t, win, padding=5, groups=3)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    s_aa = f(a * a) - mu_a ** 2
    s_bb = f(b * b) - mu_b ** 2
    s_ab = f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def gsplat_render(params, viewmat, K, width, height, sh_degree, mode="RGB"):
    import torch
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=colors, viewmats=viewmat, Ks=K,
        width=width, height=height, sh_degree=sh_degree, near_plane=0.01, far_plane=1e10, packed=False,
        render_mode=mode)
    return out[0], alpha[0], info


def scene_scale(c2ws: np.ndarray, points: np.ndarray) -> float:
    centers = c2ws[:, :3, 3]
    cam_ext = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) if len(centers) > 1 else 0.0
    d = np.linalg.norm(points - centers[0], axis=1)
    return max(cam_ext * 1.1, 0.25 * float(np.median(d)) if len(d) else 1.0, 1e-3)


def train(images: list, c2ws: np.ndarray, Ks: np.ndarray, weights, points: np.ndarray, colors: np.ndarray,
          cfg: TrainConfig | None = None, device="cuda", render_fn=None, progress=None, cancel=None):
    """Optimise Gaussians. ``images``: list of (H,W,3) uint8 arrays (sizes may differ)."""
    import torch
    import torch.nn.functional as F

    cfg = cfg or TrainConfig()
    render_fn = render_fn or gsplat_render
    torch.manual_seed(cfg.seed)
    params = init_params(points, colors, cfg, device)
    scale = scene_scale(c2ws, points)
    opt = {
        "means": torch.optim.Adam([params["means"]], lr=cfg.lr_means * scale, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=cfg.lr_scales, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=cfg.lr_quats, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=cfg.lr_opacities, eps=1e-15),
        "sh0": torch.optim.Adam([params["sh0"]], lr=cfg.lr_sh0, eps=1e-15),
        "shN": torch.optim.Adam([params["shN"]], lr=cfg.lr_shN, eps=1e-15),
    }
    means_sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / max(cfg.steps, 1)))
    strategy = state = None
    if cfg.densify:
        from gsplat.strategy import DefaultStrategy

        strategy = DefaultStrategy(verbose=False, refine_stop_iter=int(cfg.steps * 0.75),
                                   reset_every=max(3000, cfg.steps // 4))
        strategy.check_sanity(params, opt)
        state = strategy.initialize_state(scene_scale=scale)
    gts = [torch.from_numpy(np.ascontiguousarray(im)).to(device) for im in images]
    viewmats = torch.linalg.inv(torch.as_tensor(np.asarray(c2ws), dtype=torch.float32)).to(device)
    Kt = torch.as_tensor(np.asarray(Ks), dtype=torch.float32).to(device)
    w = torch.as_tensor(np.asarray(weights, np.float64) / np.sum(weights), dtype=torch.float32)
    g = torch.Generator().manual_seed(cfg.seed)
    history = []
    for step in range(cfg.steps):
        if cancel and cancel():
            raise KeyboardInterrupt("cancelled")
        i = int(torch.multinomial(w, 1, generator=g))
        gt = gts[i].float() / 255.0
        K = Kt[i].clone()
        h, wd = gt.shape[:2]
        if step < cfg.coarse_until * cfg.steps and min(h, wd) >= 64:
            h2, w2 = h // 2, wd // 2
            gt = F.interpolate(gt.permute(2, 0, 1)[None], size=(h2, w2), mode="area")[0].permute(1, 2, 0)
            K[:2] *= torch.tensor([[w2 / wd], [h2 / h]], device=K.device)
            h, wd = h2, w2
        sh_deg = min(step // 1000, cfg.sh_degree)
        rgb, alpha, info = render_fn(params, viewmats[i:i + 1], K[None], wd, h, sh_deg)
        if strategy is not None:
            strategy.step_pre_backward(params, opt, state, step, info)
        rgb = rgb[..., :3]
        l1 = (rgb - gt).abs().mean()
        loss = l1
        if cfg.ssim_weight > 0:
            loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (
                1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        s = params["scales"].exp()
        if cfg.scale_reg > 0:
            ratio = s.max(1).values / s.min(1).values.clamp_min(1e-12)
            loss = loss + cfg.scale_reg * (ratio - cfg.max_aniso).clamp_min(0).mean()
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * params["opacities"].sigmoid().mean()
        loss.backward()
        if strategy is not None:
            if len(params["means"]) >= cfg.max_gaussians and strategy.refine_stop_iter > step:
                strategy.refine_stop_iter = step   # cap reached: stop growing, keep pruning off
            strategy.step_post_backward(params, opt, state, step, info, packed=False)
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        means_sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.append((step, float(loss.detach())))
            if progress:
                progress((step + 1) / cfg.steps, f"training splats {step + 1}/{cfg.steps} · "
                                                 f"{len(params['means']):,} splats · loss {float(loss.detach()):.4f}")
    return params, history


def visibility_counts(params, c2ws, Ks, sizes, view_ids, render_depth=None, tol: float = 0.05):
    """For every splat, in how many of the given views it is visible (inside the
    frame, in front of the camera, and not hidden behind the rendered surface)."""
    import torch

    means = params["means"].detach()
    counts = torch.zeros(len(means), dtype=torch.int32, device=means.device)
    for v in view_ids:
        c2w = torch.as_tensor(np.asarray(c2ws[v]), dtype=torch.float32, device=means.device)
        K = torch.as_tensor(np.asarray(Ks[v]), dtype=torch.float32, device=means.device)
        w, h = sizes[v]
        pc = (means - c2w[:3, 3]) @ c2w[:3, :3]
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z.clamp_min(1e-9) + K[0, 2]
        y = K[1, 1] * pc[:, 1] / z.clamp_min(1e-9) + K[1, 2]
        vis = (z > 1e-6) & (u >= 0) & (u < w) & (y >= 0) & (y < h)
        if render_depth is not None:
            dep = render_depth(v)                        # (H,W) expected depth of the rendered surface
            ui = u.long().clamp(0, w - 1)
            yi = y.long().clamp(0, h - 1)
            surf = dep[yi, ui]
            vis &= (z <= surf * (1 + tol) + 1e-6) | ~torch.isfinite(surf) | (surf <= 0)
        counts += vis.int()
    return counts


def provenance_codes(real_counts, any_generated: bool):
    """0 observed (>= 2 real views), 1 inferred (1 real view), 2 generative (only generated views)."""
    import torch

    codes = torch.full_like(real_counts, 1)
    codes[real_counts >= 2] = 0
    codes[real_counts == 0] = 2 if any_generated else 1
    return codes.to(torch.uint8)


def write_ply(params, path: str, provenance=None) -> int:
    """INRIA 3DGS PLY (binary little-endian), optional 'provenance' uchar property."""
    import torch

    with torch.no_grad():
        means = params["means"].float().cpu().numpy()
        n = len(means)
        sh0 = params["sh0"].float().cpu().numpy().reshape(n, 3)
        shN = params["shN"].float().cpu().numpy()          # (N, K, 3)
        rest = shN.transpose(0, 2, 1).reshape(n, -1)       # INRIA: channel-major
        op = params["opacities"].float().cpu().numpy()
        sc = params["scales"].float().cpu().numpy()
        q = torch.nn.functional.normalize(params["quats"].float(), dim=1).cpu().numpy()
    ok = np.isfinite(means).all(1) & np.isfinite(sc).all(1) & np.isfinite(op)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(rest.shape[1])]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dt = [(p, "<f4") for p in props] + ([("provenance", "u1")] if provenance is not None else [])
    arr = np.zeros(int(ok.sum()), dtype=dt)
    arr["x"], arr["y"], arr["z"] = means[ok].T
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh0[ok].T
    for i in range(rest.shape[1]):
        arr[f"f_rest_{i}"] = rest[ok, i]
    arr["opacity"] = op[ok]
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = sc[ok].T
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = q[ok].T
    if provenance is not None:
        arr["provenance"] = np.asarray(provenance.cpu().numpy() if hasattr(provenance, "cpu") else provenance)[ok]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    header += [f"property {'float' if t == '<f4' else 'uchar'} {name}" for name, t in dt]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes())
    return len(arr)


def psnr(a, b) -> float:
    mse = float(((a.float() - b.float()) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))
