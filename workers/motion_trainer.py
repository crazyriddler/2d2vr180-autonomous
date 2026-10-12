"""Smooth 4D motion for the moving parts of a video (runtime 'recon3d-cu124').

The per-frame splats of what moves (video4d_worker.frame_splats) are a flip-book: every frame is a new set
of splats, so a person flickers between frames and nothing exists between two frames. Here the moving
parts become Gaussians that live for a while and move (the Spacetime-Gaussian parametrisation used by
scene4d / .splatv):

    position(t) = mean + v dt + a dt^2,   rotation(t) = quat + omega dt,
    opacity(t)  = opacity * exp(-(dt / t_scale)^2),          dt = t - t_center

Gaussians are seeded from the moving splats of every ``keyframe_stride``-th frame, with a velocity from the
nearest point of the next keyframe, and then optimised on every frame together with the (fixed) trained
background: colour on the moving pixels (the rest weighted less), depth on the moving pixels, local
rigidity (neighbours keep their distances while moving) and small accelerations. The result plays
smoothly at any frame rate and needs far fewer splats than one set per frame.

Plain PyTorch around a rasteriser (gsplat in the runtime, a reference renderer in the tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SH_C0 = 0.28209479177387814


@dataclass
class MotionConfig:
    steps: int = 4000
    keyframe_stride: int = 2
    lr_means: float = 1.6e-4        # times scene scale
    lr_motion: float = 1e-2         # times scene scale / keyframe interval (velocity, acceleration)
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_omega: float = 1e-4
    lr_opacities: float = 5e-2
    lr_colors: float = 2.5e-3
    lr_tscale: float = 1e-2
    depth_weight: float = 0.1
    still_weight: float = 0.2       # colour loss weight outside the moving pixels
    rigid_weight: float = 0.5
    rigid_samples: int = 4096
    accel_weight: float = 0.01
    knn: int = 8
    seed: int = 0
    log_every: int = 250


def _nearest(src, dst):
    """Index and distance of the nearest dst point for every src point (numpy)."""
    try:
        from sklearn.neighbors import NearestNeighbors

        d, i = NearestNeighbors(n_neighbors=1).fit(dst).kneighbors(src)
        return i[:, 0], d[:, 0]
    except ImportError:
        import torch

        a, b = torch.as_tensor(src), torch.as_tensor(dst)
        out_i, out_d = [], []
        for s in range(0, len(a), 4096):
            d = torch.cdist(a[s:s + 4096], b)
            v, i = d.min(1)
            out_i.append(i)
            out_d.append(v)
        return torch.cat(out_i).numpy(), torch.cat(out_d).numpy()


def _clusters(points, eps):
    """Labels of spatially separate groups (people, animals...) in a point cloud; one group without sklearn."""
    try:
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=eps, min_samples=4).fit(points).labels_
    except ImportError:
        return np.zeros(len(points), np.int64)


def follow(src, dst, iters=15):
    """Where each point of ``src`` went in ``dst``: every separate group (person, animal...) is first moved
    as a whole - from its centre to the centre of the nearest group of dst, then refined by
    translation-only ICP (points matched to their nearest dst points, mean offset applied, repeatedly) -
    and then each point takes its nearest point from there. Returns the displacement per point."""
    if not len(src) or not len(dst):
        return np.zeros_like(src)
    spacing = _knn_spacing(src)
    labels = _clusters(src, 4 * spacing)
    dlab = _clusters(dst, 4 * spacing)
    dcent = np.stack([dst[dlab == k].mean(0) for k in np.unique(dlab)])
    disp = np.zeros_like(src)
    for lab in np.unique(labels):
        sel = labels == lab
        pts = src[sel]
        c = pts.mean(0)
        t = (dcent[np.argmin(np.linalg.norm(dcent - c, axis=1))] - c).astype(np.float32)
        for _ in range(iters):
            i, _ = _nearest(pts + t, dst)
            step = (dst[i] - (pts + t)).mean(0)
            t = t + step
            if np.linalg.norm(step) < 1e-4 * max(spacing, 1e-6):
                break
        i, _ = _nearest(pts + t, dst)
        local = dst[i] - (pts + t)
        local[np.linalg.norm(local, axis=1) > 3 * spacing] = 0       # far matches are not the same surface
        disp[sel] = t + local
    return disp


def _knn_spacing(points):
    """Median distance between neighbouring points."""
    if len(points) < 2:
        return 1.0
    sub = points[np.random.default_rng(0).permutation(len(points))[:4000]]
    i = _knn(sub, 1)[:, 0]
    return float(np.median(np.linalg.norm(sub - sub[i], axis=1))) or 1e-3


def _knn(points, k):
    """k nearest neighbours (excluding the point itself) of every point (numpy)."""
    k = min(k, len(points) - 1)
    if k < 1:
        return np.zeros((len(points), 0), np.int64)
    try:
        from sklearn.neighbors import NearestNeighbors

        _, i = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
        return i[:, 1:]
    except ImportError:
        import torch

        p = torch.as_tensor(points)
        out = []
        for s in range(0, len(p), 2048):
            out.append(torch.cdist(p[s:s + 2048], p).topk(k + 1, largest=False).indices[:, 1:])
        return torch.cat(out).numpy()


def seed_motion(keyframes, cfg: MotionConfig, max_speed: float | None = None):
    """Initial moving Gaussians from per-keyframe splats.

    ``keyframes``: list of dicts (means, scales, quats, opacity, colors as numpy; "t": the keyframe's time)
    in time order. Each Gaussian starts at its keyframe's time, lives about one keyframe interval either
    side and moves the way its group moved between the keyframes before and after (``follow``: a first guess
    the optimisation corrects).
    Returns numpy arrays and the k-nearest-neighbour graph (within each keyframe)."""
    ts = [kf["t"] for kf in keyframes]
    gap = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    parts, nbrs, off = [], [], 0
    for j, kf in enumerate(keyframes):
        m = kf["means"].astype(np.float32)
        n = len(m)
        vel = np.zeros((n, 3), np.float32)
        if n and len(keyframes) > 1:
            est = []
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(keyframes) and len(keyframes[jj]["means"]):
                    v = follow(m, keyframes[jj]["means"]) / (keyframes[jj]["t"] - kf["t"])
                    if max_speed:
                        v[np.linalg.norm(v, axis=1) > max_speed] = 0
                    est.append(v)
            if est:
                vel = np.mean(est, 0).astype(np.float32)
        parts.append({"means": m, "scales": kf["scales"].astype(np.float32), "quats": kf["quats"].astype(np.float32),
                      "opacity": kf["opacity"].astype(np.float32), "colors": kf["colors"].astype(np.float32),
                      "vel": vel, "t_center": np.full(n, kf["t"], np.float32),
                      "t_scale": np.full(n, 0.6 * gap, np.float32)})
        nbrs.append(_knn(m, cfg.knn) + off if n > 1 else np.zeros((n, 0), np.int64))
        off += n
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    width = max((x.shape[1] for x in nbrs), default=0)
    graph = np.full((off, width), -1, np.int64)
    r = 0
    for x in nbrs:
        graph[r:r + len(x), :x.shape[1]] = x
        r += len(x)
    return out, graph


def to_params(seed, device):
    import torch

    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)  # noqa: E731
    op = np.clip(seed["opacity"], 1e-4, 1 - 1e-4)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(seed["means"])),
        "vel": torch.nn.Parameter(t(seed["vel"])),
        "acc": torch.nn.Parameter(torch.zeros(len(seed["means"]), 3, device=device)),
        "scales": torch.nn.Parameter(t(np.log(np.maximum(seed["scales"], 1e-7)))),
        "quats": torch.nn.Parameter(t(seed["quats"])),
        "omega": torch.nn.Parameter(torch.zeros(len(seed["means"]), 4, device=device)),
        "opacities": torch.nn.Parameter(t(np.log(op / (1 - op)))),
        "sh0": torch.nn.Parameter(t((np.clip(seed["colors"], 0, 1) - 0.5) / SH_C0)[:, None, :]),
        "log_tscale": torch.nn.Parameter(t(np.log(seed["t_scale"]))),
    }), t(seed["t_center"])


def state_at(params, t_center, t, idx=None):
    """Positions, rotations, log-scales, opacity logits and base colour of the Gaussians (``idx``) at t."""
    import torch

    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    dt = (t - sel(t_center))[:, None]
    means = sel(params["means"]) + sel(params["vel"]) * dt + sel(params["acc"]) * dt ** 2
    quats = sel(params["quats"]) + sel(params["omega"]) * dt
    op = torch.sigmoid(sel(params["opacities"])) * torch.exp(-(dt[:, 0] / sel(params["log_tscale"]).exp()) ** 2)
    return means, quats, sel(params["scales"]), torch.logit(op.clamp(1e-6, 1 - 1e-6)), sel(params["sh0"])


def train_motion(params, t_center, graph, images, c2ws, Ks, times, moving, depths, cfg: MotionConfig,
                 background=None, render_fn=None, device="cuda", progress=None, cancel=None):
    """Optimise the moving Gaussians on every frame.

    ``images``: (H,W,3) uint8 per frame; ``moving``: (H,W) bool per frame; ``depths``: (H,W) aligned depth per
    frame (0 = unknown); ``times``: scene time of every frame; ``background``: dict of fixed tensors
    (means, quats, scales (log), opacities (logit), sh0, shN) drawn with the moving Gaussians, or None."""
    import torch

    if render_fn is None:
        from splat_trainer import gsplat_render as render_fn
    from splat_trainer import ssim

    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    c2ws_np = np.asarray(c2ws)
    centers = c2ws_np[:, :3, 3]
    pts = params["means"].detach().cpu().numpy()
    scale = max(float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) if len(centers) > 1 else 0.0,
                0.25 * float(np.median(np.linalg.norm(pts - centers[0], axis=1))) if len(pts) else 1.0, 1e-3)
    tc_np = np.unique(t_center.detach().cpu().numpy())
    gap = float(np.median(np.diff(tc_np))) if len(tc_np) > 1 else 1.0
    opt = torch.optim.Adam([
        {"params": [params["means"]], "lr": cfg.lr_means * scale},
        {"params": [params["vel"]], "lr": cfg.lr_motion * scale / gap},
        {"params": [params["acc"]], "lr": cfg.lr_motion * scale / gap ** 2},
        {"params": [params["scales"]], "lr": cfg.lr_scales},
        {"params": [params["quats"]], "lr": cfg.lr_quats},
        {"params": [params["omega"]], "lr": cfg.lr_omega},
        {"params": [params["opacities"]], "lr": cfg.lr_opacities},
        {"params": [params["sh0"]], "lr": cfg.lr_colors},
        {"params": [params["log_tscale"]], "lr": cfg.lr_tscale},
    ], eps=1e-15)
    viewmats = torch.linalg.inv(torch.as_tensor(c2ws_np, dtype=torch.float32)).to(device)
    Kt = torch.as_tensor(np.asarray(Ks), dtype=torch.float32).to(device)
    gts = [torch.from_numpy(np.ascontiguousarray(im)).to(device) for im in images]
    wts = [torch.where(torch.from_numpy(np.ascontiguousarray(m)).to(device), 1.0, cfg.still_weight) for m in moving]
    dps = [torch.from_numpy(np.where(m, d, 0).astype(np.float32)).to(device) for m, d in zip(moving, depths)]
    graph_t = torch.as_tensor(graph, device=device)
    has_nbr = (graph_t >= 0).any(1).nonzero()[:, 0] if graph.shape[1] else torch.zeros(0, dtype=torch.long)
    sh_rest = background["shN"].shape[1] if background is not None else 0
    history = []
    for step in range(cfg.steps):
        if cancel and cancel():
            raise KeyboardInterrupt("cancelled")
        i = int(torch.randint(len(images), (1,), generator=g))
        t = float(times[i])
        with torch.no_grad():
            live = ((t - t_center).abs() < 3 * params["log_tscale"].exp()).nonzero()[:, 0]
        means, quats, scales, ops, sh0 = state_at(params, t_center, t, live)
        p = {"means": means, "quats": quats, "scales": scales, "opacities": ops, "sh0": sh0,
             "shN": torch.zeros(len(live), sh_rest, 3, device=device)}
        if background is not None:
            p = {k: torch.cat([background[k], p[k]]) for k in p}
        h, w = gts[i].shape[:2]
        out, alpha, _ = render_fn(p, viewmats[i:i + 1], Kt[i:i + 1], w, h, 0 if background is None else
                                  int(round(math.sqrt(sh_rest + 1))) - 1, mode="RGB+ED")
        rgb, dep = out[..., :3], out[..., 3]
        gt = gts[i].float() / 255
        wm = wts[i]
        l1 = ((rgb - gt).abs().mean(-1) * wm).sum() / wm.sum()
        gt_s = gt * (wm[..., None] > 0.5) + rgb.detach() * (wm[..., None] <= 0.5)
        loss = 0.8 * l1 + 0.2 * (1 - ssim(rgb.permute(2, 0, 1)[None], gt_s.permute(2, 0, 1)[None]))
        dm = dps[i]
        valid = (dm > 0) & (alpha[..., 0] > 0.5)
        if cfg.depth_weight > 0 and bool(valid.any()):
            loss = loss + cfg.depth_weight * ((dep[valid] - dm[valid]).abs() / dm[valid]).mean()
        if cfg.rigid_weight > 0 and len(has_nbr):
            pick = has_nbr[torch.randint(len(has_nbr), (min(cfg.rigid_samples, len(has_nbr)),), generator=g)
                           .to(has_nbr.device)]
            nb = graph_t[pick]
            ok = nb >= 0
            nb = nb.clamp_min(0)
            a_t = state_at(params, t_center, t, pick)[0]
            b_t = state_at(params, t_center, t, nb.reshape(-1))[0].reshape(*nb.shape, 3)
            d_t = (a_t[:, None] - b_t).norm(dim=-1)
            d_0 = (params["means"][pick][:, None] - params["means"][nb]).norm(dim=-1).detach()
            loss = loss + cfg.rigid_weight * (((d_t - d_0).abs() / (d_0 + 1e-6)) * ok).sum() / ok.sum().clamp_min(1)
        if cfg.accel_weight > 0:
            loss = loss + cfg.accel_weight * params["acc"][live].square().sum(1).mean() / scale ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.append((step, float(loss.detach())))
            if progress:
                progress((step + 1) / cfg.steps, f"moving parts in 4D {step + 1}/{cfg.steps} · loss "
                                                 f"{float(loss.detach()):.4f}")
    return history


def export_motion(params, t_center, min_opacity=0.02):
    """Arrays in the scene4d.DynamicSplats layout (motion = [v, a, 0]); nearly transparent Gaussians are
    dropped."""
    import torch

    with torch.no_grad():
        op = torch.sigmoid(params["opacities"]).cpu().numpy()
        keep = op >= min_opacity
        n = int(keep.sum())
        q = torch.nn.functional.normalize(params["quats"], dim=1).cpu().numpy()[keep]
        motion = np.zeros((n, 9), np.float32)
        motion[:, 0:3] = params["vel"].cpu().numpy()[keep]
        motion[:, 3:6] = params["acc"].cpu().numpy()[keep]
        return {"means": params["means"].cpu().numpy()[keep].astype(np.float32),
                "scales": params["scales"].exp().cpu().numpy()[keep].astype(np.float32),
                "quats": q.astype(np.float32),
                "opacity": op[keep].astype(np.float32),
                "colors": np.clip(params["sh0"][:, 0].cpu().numpy()[keep] * SH_C0 + 0.5, 0, 1).astype(np.float32),
                "t_center": t_center.cpu().numpy()[keep].astype(np.float32),
                "t_scale": params["log_tscale"].exp().cpu().numpy()[keep].astype(np.float32),
                "motion": motion, "omega": params["omega"].cpu().numpy()[keep].astype(np.float32)}
