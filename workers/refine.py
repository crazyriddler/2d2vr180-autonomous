"""The photo's splat fitted to the photo itself (2.0.0-rc7) - runtime 'photo-cu128' (gsplat).

SHARP predicts every splat in one go; seen from the photo's own camera its render is close to the photo but
not equal (softer, small colour shifts, thin gaps between splats). Here the splat is rendered from that
camera and its colours, opacities and sizes are adjusted for a few hundred steps until the render matches
the photo (L1 + structural similarity, as 3D Gaussian Splatting trains). Only the photo is used - nothing is
invented - and positions never move, so the measured geometry stays exactly as it is. Sizes may change by
at most ~35 % and opacities are kept near their start, so what is seen from other places does not degrade.
The result is kept only when the fit to the photo really improved.
"""

import numpy as np

SH_C0 = 0.28209479177387814


def ssim(a, b, torch, win=11, sigma=1.5):
    """Mean structural similarity of two (1, 3, H, W) images in [0, 1]."""
    import torch.nn.functional as F

    x = torch.arange(win, dtype=a.dtype, device=a.device) - win // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :]).expand(3, 1, win, win).contiguous()
    f = lambda t: F.conv2d(t, k, padding=win // 2, groups=3)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    va = f(a * a) - mu_a ** 2
    vb = f(b * b) - mu_b ** 2
    cov = f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean()


def gsplat_render(means, quats, scales, opacities, sh, viewmat, K, width, height, sh_degree):
    """gsplat's rasteriser (as render_worker.py): (rgb (H, W, 3), alpha (H, W))."""
    from gsplat import rasterization

    out, alpha, _ = rasterization(means, quats, scales, opacities, sh, viewmat[None], K[None], width, height,
                                  sh_degree=sh_degree, near_plane=0.01, render_mode="RGB")
    return out[0, ..., :3], alpha[0, ..., 0]


def _read(path):
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    st = lambda *ks: np.stack([np.asarray(v[k], np.float32) for k in ks], 1)  # noqa: E731
    rest = sorted([k for k in names if k.startswith("f_rest_")], key=lambda s: int(s[7:]))
    return ply, {"means": st("x", "y", "z"), "quats": st("rot_0", "rot_1", "rot_2", "rot_3"),
                 "log_scales": st("scale_0", "scale_1", "scale_2"), "logit": np.asarray(v["opacity"], np.float32),
                 "dc": st("f_dc_0", "f_dc_1", "f_dc_2"),
                 "rest": st(*rest).reshape(len(v), 3, -1).transpose(0, 2, 1) if rest else None}


def refine_ply(ply_in, ply_out, image, fx, size, torch, device, iters=300, max_side=2048, render=None,
               log=lambda m: None, max_scale_change=0.3):
    """Fit the splat in ``ply_in`` (camera frame of the photo, principal point at the centre of ``size`` =
    (W, H) pixels with focal length ``fx``) to ``image`` (H', W', 3 uint8, the photo at any size) and write
    ``ply_out`` when the fit improved. Returns stats."""
    from PIL import Image

    render = render or gsplat_render
    ply, p = _read(ply_in)
    W, H = int(size[0]), int(size[1])
    s = min(1.0, max_side / max(W, H))
    w, h = max(8, round(W * s)), max(8, round(H * s))
    target = np.asarray(Image.fromarray(np.asarray(image, np.uint8)).resize((w, h), Image.LANCZOS), np.float32) / 255
    tgt = torch.as_tensor(target, device=device)
    K = torch.tensor([[fx * w / W, 0, w / 2], [0, fx * h / H, h / 2], [0, 0, 1]], dtype=torch.float32, device=device)
    view = torch.eye(4, dtype=torch.float32, device=device)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
    means, quats, ls0 = T(p["means"]), T(p["quats"]), T(p["log_scales"])
    rest = T(p["rest"]) if p["rest"] is not None else None
    degree = int(round(np.sqrt(1 + rest.shape[1]))) - 1 if rest is not None else 0
    dc = torch.nn.Parameter(T(p["dc"]))
    logit0 = T(p["logit"])
    logit = torch.nn.Parameter(logit0.clone())
    dscale = torch.nn.Parameter(torch.zeros_like(ls0))
    opt = torch.optim.Adam([{"params": [dc], "lr": 0.01}, {"params": [logit], "lr": 0.05},
                            {"params": [dscale], "lr": 0.02}], eps=1e-15)

    def draw():
        sh = dc[:, None, :] if rest is None else torch.cat([dc[:, None, :], rest], 1)
        scales = torch.exp(ls0 + max_scale_change * torch.tanh(dscale))
        rgb, _ = render(means, quats, scales, torch.sigmoid(logit), sh, view, K, w, h, degree)
        return rgb

    def loss_of(rgb):
        a = rgb.clamp(0, 1).permute(2, 0, 1)[None]
        b = tgt.permute(2, 0, 1)[None]
        return 0.8 * (a - b).abs().mean() + 0.2 * (1 - ssim(a, b, torch))

    def psnr(rgb):
        mse = float(((rgb.clamp(0, 1) - tgt) ** 2).mean())
        return 10 * np.log10(1.0 / max(mse, 1e-10))

    with torch.no_grad():
        first = draw()
        before, psnr0 = float(loss_of(first)), psnr(first)
    for it in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        rgb = draw()
        loss = loss_of(rgb) + 1e-3 * ((logit - logit0) ** 2).mean()
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            return {"used": False, "reason": "the fit diverged", "iterations": it}
    with torch.no_grad():
        last = draw()
        after, psnr1 = float(loss_of(last)), psnr(last)
    stats = {"iterations": int(iters), "render_size": [w, h], "loss_before": round(before, 5),
             "loss_after": round(after, 5), "psnr_before": round(psnr0, 2), "psnr_after": round(psnr1, 2)}
    if not (np.isfinite(after) and after < 0.99 * before):
        stats.update(used=False, reason="no clear improvement")
        return stats
    v = ply["vertex"].data
    d = dc.detach().cpu().numpy()
    lg = logit.detach().cpu().numpy()
    sc = (ls0 + max_scale_change * torch.tanh(dscale)).detach().cpu().numpy()
    for i in range(3):
        v[f"f_dc_{i}"] = d[:, i]
        v[f"scale_{i}"] = sc[:, i]
    v["opacity"] = lg
    ply.write(ply_out)
    stats.update(used=True, scale_change_p90=round(float(np.quantile(np.abs(sc - p["log_scales"]), 0.9)), 4))
    log(f"fitted to the photo: {stats['psnr_before']:.1f} -> {stats['psnr_after']:.1f} dB PSNR from the photo's "
        f"camera ({iters} steps at {w}x{h})")
    return stats
