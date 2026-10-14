"""People's geometry with Sapiens2 (Meta, ICLR 2026) - runtime 'photo-cu128' (rc58).

General monocular depth (MoGe) flattens people: noses, chins and the sides of a face come out shallow,
and its scale for a person is a guess from the whole scene. Sapiens2 is trained on a billion human images
and synthetic scans; its pointmap model predicts each pixel's 3D position for people, with the fine
shape of faces and bodies. This module:
  1. finds the people (Sapiens2 body-part segmentation, class 0 = background);
  2. predicts their pointmap on a crop around them at the network's 1024x768 (Sapiens2 pointmap);
  3. turns it into depth with the photo's real lens (the pointmap is in a canonical 768 px focal space);
  4. puts that shape into MoGe's depth inside the people (fuse_people): the person's shape from Sapiens2,
     its distance from the scene's own depth, blended over a few pixels inside the outline.
It also reports how far MoGe's distance to the people is from Sapiens2's (a second real-world scale cue).

Only the backbone is taken from Sapiens2 (sapiens2_backbone.py, vendored verbatim); the two small heads
below are inference-only copies of sapiens.dense PointmapHead / SegHead with the same parameter names,
checked against the originals (identical outputs on the same weights).
"""

import numpy as np

EMBED = {"sapiens2_0.4b": 1024, "sapiens2_0.8b": 1280, "sapiens2_1b": 1536, "sapiens2_5b": 2432}
NET_H, NET_W = 1024, 768
CANONICAL_F = 768.0
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def _heads():
    import torch.nn as nn

    class PointmapHead(nn.Module):
        def __init__(self, c, up=(1536, 768, 512, 256), conv=(64, 32, 16), ks=(3, 3, 3), sconv=(1536, 512, 128),
                     sfinal=(6144, 512, 128, 1)):
            super().__init__()
            self.input_conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.InstanceNorm2d(c), nn.SiLU(inplace=True))
            blocks, cur = [], c
            for out in up:
                blocks.append(nn.Sequential(nn.Conv2d(cur, out * 4, 3, padding=1), nn.PixelShuffle(2),
                                            nn.InstanceNorm2d(out), nn.SiLU(inplace=True)))
                cur = out
            self.upsample_blocks = nn.Sequential(*blocks)
            layers = []
            for out, k in zip(conv, ks):
                layers += [nn.Conv2d(cur, out, k, padding=(k - 1) // 2), nn.InstanceNorm2d(out), nn.SiLU(inplace=True)]
                cur = out
            self.conv_layers = nn.Sequential(*layers)
            self.conv_pointmap = nn.Conv2d(cur, 3, kernel_size=1)
            sl, cur = [], c
            for out in sconv:
                sl += [nn.Conv2d(cur, out, 1, stride=2), nn.InstanceNorm2d(out), nn.SiLU(inplace=True)]
                cur = out
            self.scale_conv_layers = nn.Sequential(*sl)
            fl, cur = [nn.Flatten()], sfinal[0]
            for i, out in enumerate(sfinal[1:]):
                fl.append(nn.Linear(cur, out))
                if i < len(sfinal) - 2:
                    fl.append(nn.SiLU())
                cur = out
            self.scale_final_layer = nn.Sequential(*fl)

        def forward(self, x):
            p = self.conv_pointmap(self.conv_layers(self.upsample_blocks(self.input_conv(x))))
            return p, self.scale_final_layer(self.scale_conv_layers(x))

    class SegHead(nn.Module):
        def __init__(self, c, deconv=(512, 256, 128, 64), conv=(64, 64), classes=29):
            super().__init__()
            layers, cur = [], c
            for out in deconv:
                layers += [nn.ConvTranspose2d(cur, out, 4, stride=2, padding=1, output_padding=0, bias=False),
                           nn.InstanceNorm2d(out), nn.SiLU(inplace=True)]
                cur = out
            self.deconv_layers = nn.Sequential(*layers)
            layers = []
            for out in conv:
                layers += [nn.Conv2d(cur, out, 1), nn.InstanceNorm2d(out), nn.SiLU(inplace=True)]
                cur = out
            self.conv_layers = nn.Sequential(*layers)
            self.conv_seg = nn.Conv2d(cur, classes, kernel_size=1)

        def forward(self, x):
            return self.conv_seg(self.conv_layers(self.deconv_layers(x)))

    return PointmapHead, SegHead


def arch_of(path: str) -> str:
    """'sapiens2_1b' from a checkpoint file name such as sapiens2_1b_pointmap.safetensors."""
    import os

    name = os.path.basename(path).lower()
    for a in sorted(EMBED, key=len, reverse=True):
        if name.startswith(a + "_") or name.startswith(a + "."):
            return a
    return "sapiens2_1b"


def load(path: str, task: str, device, torch):
    """Backbone + head for 'pointmap' or 'seg' from a released Sapiens2 safetensors checkpoint."""
    import torch.nn as nn
    from safetensors.torch import load_file
    from sapiens2_backbone import Sapiens2

    arch = arch_of(path)
    c = EMBED[arch]
    PointmapHead, SegHead = _heads()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Sapiens2(arch=arch, img_size=(NET_H, NET_W), patch_size=16, final_norm=True,
                                     use_tokenizer=False, with_cls_token=True, out_type="featmap")
            self.decode_head = PointmapHead(c) if task == "pointmap" else SegHead(c)

        def forward(self, x):
            return self.decode_head(self.backbone(x)[0])

    net = Net()
    sd = load_file(path, device="cpu")
    for p in ("model.", "module.", "estimator."):
        if all(k.startswith(p) for k in sd):
            sd = {k[len(p):]: v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys]
    if len(missing) > 0.02 * len(net.state_dict()):
        raise RuntimeError(f"Sapiens2 {task} checkpoint does not match ({len(missing)} missing weights, e.g. "
                           f"{missing[:3]})")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return net.to(device=device, dtype=dtype).eval(), arch


def _to_input(img, torch, device, dtype):
    x = (img.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)[None].to(device=device, dtype=dtype)


def people_mask(net, img, torch, device):
    """Bool (H,W): pixels of people (any body part)."""
    import torch.nn.functional as F
    from PIL import Image

    h, w = img.shape[:2]
    dtype = next(net.parameters()).dtype
    x = np.asarray(Image.fromarray(img).resize((NET_W, NET_H), Image.BILINEAR))     # stretched, as trained
    with torch.no_grad():
        logits = net(_to_input(x, torch, device, dtype)).float()
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return (logits.argmax(1)[0] > 0).cpu().numpy()


def crop_box(mask, margin=0.12):
    """A box around the people with a margin, grown to the network's 3:4 aspect. None without people."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    y0, y1, x0, x1 = max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)
    bh, bw = y1 - y0, x1 - x0
    if bw * NET_H < bh * NET_W:          # too narrow: widen
        need = int(np.ceil(bh * NET_W / NET_H)) - bw
        x0, x1 = max(0, x0 - need // 2), min(w, x1 + need - need // 2)
    else:                                 # too wide: make taller
        need = int(np.ceil(bw * NET_H / NET_W)) - bh
        y0, y1 = max(0, y0 - need // 2), min(h, y1 + need - need // 2)
    return int(y0), int(y1), int(x0), int(x1)


def pointmap_depth(net, img, box, f_px, torch, device):
    """Sapiens2's depth of a crop (rows y0:y1, cols x0:x1 of img) in metres for a lens of f_px pixels, and
    its own estimate of that lens (pixels), from the canonical-focal pointmap."""
    import torch.nn.functional as F
    from PIL import Image

    y0, y1, x0, x1 = box
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = min(NET_W / cw, NET_H / ch)
    nw, nh = int(cw * s), int(ch * s)
    rs = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.BILINEAR))
    pl, pt = (NET_W - nw) // 2, (NET_H - nh) // 2
    canvas = np.zeros((NET_H, NET_W, 3), np.uint8)
    canvas[pt:pt + nh, pl:pl + nw] = rs
    dtype = next(net.parameters()).dtype
    with torch.no_grad():
        pm, scale = net(_to_input(canvas, torch, device, dtype))
        pm = pm.float()[:, :, pt:pt + nh, pl:pl + nw]
        pm = F.interpolate(pm, size=(ch, cw), mode="bilinear", align_corners=False)
    zc = pm[0, 2].cpu().numpy().astype(np.float64)          # canonical depth: focal 768 px at network size
    f_net = f_px * s                                         # the real lens at network size
    own_f = CANONICAL_F / max(float(scale.float().item()), 1e-6) / s
    return zc * f_net / CANONICAL_F, own_f


def inner_weight(mask, ramp=6):
    """1/ramp on the outline of the mask, rising to 1 'ramp' pixels inside, 0 outside (numpy only)."""
    cur = np.asarray(mask, bool).copy()
    w = np.zeros(cur.shape, np.float32)
    for i in range(int(ramp)):
        w[cur] = (i + 1) / ramp
        p = np.pad(cur, 1)
        cur = cur & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return w


def fuse_people(depth, valid, human, hvalid, ramp=6):
    """Put the people's shape from ``human`` (depth of the same pixels, its own scale) into ``depth``.

    The shape comes from the human model; the distance (one scale factor, the median ratio over the people)
    from the scene's depth, so people stay where the scene puts them. Returns (depth, ratio): ratio =
    scene depth / human depth over the people (> 1: the scene puts them farther than the human model)."""
    both = valid & hvalid & np.isfinite(human) & (human > 0)
    if both.sum() < 200:
        return depth, None
    ratio = float(np.median(depth[both] / human[both]))
    w = inner_weight(both, ramp)
    out = np.where(both, w * human * ratio + (1 - w) * depth, depth)
    return out, ratio
