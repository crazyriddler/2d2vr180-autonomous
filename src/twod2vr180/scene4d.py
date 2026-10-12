"""4D scenes: a static background plus splats that exist or move only for part of the time.

Time is normalised to [0, 1] over the clip; frame k of F is centred at t = (k + 0.5) / F. Every
time-dependent splat follows the Spacetime-Gaussian parametrisation used by splaTV (MIT):

    position(t) = mean + m[0:3] dt + m[3:6] dt^2 + m[6:9] dt^3          dt = t - t_center
    rotation(t) = normalise(quat + omega dt)
    opacity(t)  = opacity * exp(-(dt / t_scale)^2)

so one model covers both a background trained once from all frames with the moving parts as
per-frame splats (each visible only around its own frame, motion 0) and true 4D Gaussians that
move. ``.splatv`` is splaTV's file format (16 uint32 per splat, see write_splatv) and plays in the
app's 4D viewer and in splaTV-compatible players.

Coordinates as everywhere in the app: OpenCV frame of the reference camera (scene.py).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scene import INFERRED, Camera, GaussianScene, SceneError, read_gaussian_ply, write_gaussian_ply

SPLATV_MAGIC = 0x674B
TEX_WIDTH = 4096                 # texels per row (4 texels = 16 uint32 per splat)
VISIBLE = 0.02                   # splats below this temporal opacity are not drawn (as splaTV)
STATIC_T_SCALE = 1000.0          # temporal scale of the background: visible at every t
PER_FRAME_T_SCALE = 0.45         # x frame interval: a per-frame splat is gone at the next frame (e^-4.9)


@dataclass
class DynamicSplats:
    """Splats that exist (or move) only for part of the clip."""
    means: np.ndarray                     # (N,3) float32
    scales: np.ndarray                    # (N,3) float32, linear
    quats: np.ndarray                     # (N,4) float32, (w,x,y,z)
    opacity: np.ndarray                   # (N,)
    colors: np.ndarray                    # (N,3) in [0,1]
    t_center: np.ndarray                  # (N,) in [0,1]
    t_scale: np.ndarray                   # (N,) > 0
    motion: np.ndarray | None = None      # (N,9) polynomial motion; None = still
    omega: np.ndarray | None = None       # (N,4) quaternion velocity; None = no rotation
    provenance: np.ndarray | None = None  # (N,) uint8

    def __post_init__(self) -> None:
        n = len(self.means)
        if self.provenance is None:
            self.provenance = np.full(n, INFERRED, np.uint8)
        for name, shape in (("scales", (n, 3)), ("quats", (n, 4)), ("opacity", (n,)), ("colors", (n, 3)),
                            ("t_center", (n,)), ("t_scale", (n,)), ("provenance", (n,))):
            if getattr(self, name).shape != shape:
                raise SceneError(f"dynamic {name} has shape {getattr(self, name).shape}, expected {shape}")

    def __len__(self) -> int:
        return len(self.means)

    @classmethod
    def empty(cls) -> "DynamicSplats":
        z3 = np.zeros((0, 3), np.float32)
        return cls(z3, z3.copy(), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), z3.copy(),
                   np.zeros(0, np.float32), np.ones(0, np.float32))

    @classmethod
    def concat(cls, parts: list["DynamicSplats"]) -> "DynamicSplats":
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        moving = any(p.motion is not None for p in parts)
        rot = any(p.omega is not None for p in parts)
        cat = lambda k: np.concatenate([getattr(p, k) for p in parts])  # noqa: E731
        return cls(cat("means"), cat("scales"), cat("quats"), cat("opacity"), cat("colors"), cat("t_center"),
                   cat("t_scale"),
                   np.concatenate([p.motion if p.motion is not None else np.zeros((len(p), 9), np.float32)
                                   for p in parts]) if moving else None,
                   np.concatenate([p.omega if p.omega is not None else np.zeros((len(p), 4), np.float32)
                                   for p in parts]) if rot else None,
                   cat("provenance"))

    def temporal_opacity(self, t: float) -> np.ndarray:
        dt = t - self.t_center
        return self.opacity * np.exp(-(dt / np.maximum(self.t_scale, 1e-9)) ** 2)

    def at(self, t: float, threshold: float = VISIBLE) -> GaussianScene:
        """The splats visible at time ``t`` as a plain scene (moved, rotated, faded)."""
        op = self.temporal_opacity(t)
        keep = op >= threshold
        dt = (t - self.t_center[keep])[:, None]
        means = self.means[keep].astype(np.float32)
        if self.motion is not None:
            m = self.motion[keep]
            means = (means + m[:, 0:3] * dt + m[:, 3:6] * dt ** 2 + m[:, 6:9] * dt ** 3).astype(np.float32)
        q = self.quats[keep].astype(np.float32)
        if self.omega is not None:
            q = q + self.omega[keep] * dt
        q = (q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        return GaussianScene(means, self.scales[keep].astype(np.float32), q, op[keep].astype(np.float32),
                             self.colors[keep].astype(np.float32), self.provenance[keep])

    def save(self, path: Path) -> Path:
        arrays = {k: getattr(self, k) for k in ("means", "scales", "quats", "opacity", "colors", "t_center",
                                                "t_scale", "provenance")}
        if self.motion is not None:
            arrays["motion"] = self.motion
        if self.omega is not None:
            arrays["omega"] = self.omega
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "DynamicSplats":
        with np.load(path) as d:
            g = lambda k: d[k] if k in d.files else None  # noqa: E731
            return cls(d["means"], d["scales"], d["quats"], d["opacity"], d["colors"], d["t_center"], d["t_scale"],
                       g("motion"), g("omega"), g("provenance"))


def per_frame_splats(scene: GaussianScene, frame: int, frames: int) -> DynamicSplats:
    """A frame's splats as dynamic splats visible only around that frame."""
    n = len(scene)
    return DynamicSplats(scene.means.astype(np.float32), scene.scales.astype(np.float32),
                         scene.quats.astype(np.float32), scene.opacity.astype(np.float32),
                         scene.colors.astype(np.float32), np.full(n, frame_time(frame, frames), np.float32),
                         np.full(n, PER_FRAME_T_SCALE / max(frames, 1), np.float32), provenance=scene.provenance)


def frame_time(k: int, frames: int) -> float:
    return (k + 0.5) / max(frames, 1)


@dataclass
class Scene4D:
    background: GaussianScene
    dynamic: DynamicSplats
    fps: float
    frames: int
    cameras: list[Camera] = field(default_factory=list)   # one per frame (the filming camera)
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frames / max(self.fps, 1e-6)

    def time_of(self, k: int) -> float:
        return frame_time(k, self.frames)

    def at(self, t: float) -> GaussianScene:
        """Background + what is visible at time ``t``, as one scene (cameras/scale of the background)."""
        return merge(self.background, self.dynamic.at(t))

    def frame(self, k: int) -> GaussianScene:
        return self.at(self.time_of(k))

    # ------------------------------------------------------------ storage
    def save(self, export_dir: Path, splatv: bool = True, splatv_budget: int | None = 4_000_000) -> dict:
        """export/scene.ply (the background, for any splat viewer), export/scene.splatv (the whole 4D
        scene) and export/_2d2vr180/scene4d.json + dynamic.npz (full precision, for the app)."""
        export_dir = Path(export_dir)
        ply = write_gaussian_ply(self.background, export_dir / "scene.ply")
        side = export_dir / "_2d2vr180"
        side.mkdir(parents=True, exist_ok=True)
        self.dynamic.save(side / "dynamic.npz")
        (side / "scene4d.json").write_text(json.dumps({
            "schema": "2d2vr180.scene4d/1", "fps": self.fps, "frames": self.frames,
            "duration_s": round(self.duration_s, 4), "background": "../scene.ply", "dynamic": "dynamic.npz",
            "cameras": [c.to_dict() for c in self.cameras], "meta": self.meta}, indent=1))
        out = {"scene_ply": str(ply), "scene4d": str(side / "scene4d.json")}
        if splatv:
            out["splatv"] = str(write_splatv(self, export_dir / "scene.splatv", budget=splatv_budget))
        return out

    @classmethod
    def load(cls, path: Path) -> "Scene4D":
        """From an export folder (or a file in it, e.g. its scene.ply), its scene4d.json, or a .splatv file."""
        path = Path(path)
        if path.suffix.lower() == ".splatv":
            return read_splatv(path)
        base = path.parent if path.is_file() and path.suffix != ".json" else path
        j = path if path.suffix == ".json" else base / "_2d2vr180" / "scene4d.json"
        if not j.exists() and (base / "scene.splatv").exists():
            return read_splatv(base / "scene.splatv")
        d = json.loads(j.read_text())
        bg = read_gaussian_ply((j.parent / d["background"]).resolve())
        return cls(bg, DynamicSplats.load(j.parent / d["dynamic"]), float(d["fps"]), int(d["frames"]),
                   [Camera.from_dict(c) for c in d.get("cameras", [])], d.get("meta", {}))


def is_scene4d(path: Path) -> bool:
    """A .splatv, a scene4d.json, or an export folder (or a file in it) holding a 4D scene."""
    path = Path(path)
    if path.suffix.lower() == ".splatv" or path.name == "scene4d.json":
        return True
    base = path.parent if path.is_file() else path
    return (base / "_2d2vr180" / "scene4d.json").exists()


def export_dir_of(path: Path) -> Path:
    """The export folder of a 4D scene given the folder, a file in it, or its scene4d.json."""
    path = Path(path)
    if path.name == "scene4d.json":
        return path.parent.parent
    return path.parent if path.is_file() else path


def splatv_for(path: Path) -> Path:
    """The .splatv of a 4D scene (written into its export folder when missing)."""
    path = Path(path)
    if path.suffix.lower() == ".splatv":
        return path
    out = export_dir_of(path) / "scene.splatv"
    return out if out.exists() else write_splatv(Scene4D.load(path), out)


def merge(a: GaussianScene, b: GaussianScene) -> GaussianScene:
    """One scene with the splats of both (cameras / metric scale of ``a``). Spherical harmonics beyond
    the base colour are kept for ``a`` and zero for ``b``."""
    sh = None
    if a.sh_rest is not None:
        sh = np.concatenate([a.sh_rest, np.zeros((len(b), a.sh_rest.shape[1]), np.float32)])
    return GaussianScene(np.concatenate([a.means, b.means]).astype(np.float32),
                         np.concatenate([a.scales, b.scales]).astype(np.float32),
                         np.concatenate([a.quats, b.quats]).astype(np.float32),
                         np.concatenate([a.opacity, b.opacity]).astype(np.float32),
                         np.concatenate([a.colors, b.colors]).astype(np.float32),
                         np.concatenate([a.provenance, b.provenance]).astype(np.uint8), sh,
                         list(a.cameras), a.metric_scale)


# ------------------------------------------------------------------ .splatv
def _half2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """packHalf2x16(a, b) as uint32 (a in the low 16 bits)."""
    h = np.stack([np.clip(a, -65000, 65000), np.clip(b, -65000, 65000)], 1).astype("<f2")
    return np.ascontiguousarray(h).view("<u4")[:, 0]


def _unhalf2(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.ascontiguousarray(u.astype("<u4")).view("<f2").reshape(-1, 2).astype(np.float32)
    return h[:, 0], h[:, 1]


def _importance_subset(n: int, weight: np.ndarray, budget: int) -> np.ndarray:
    if n <= budget:
        return np.arange(n)
    return np.sort(np.argsort(-weight)[:budget])


def splatv_cameras(cams: list[Camera]) -> list[dict]:
    """Cameras in splaTV's convention (3DGS cameras.json: camera-to-world rotation rows + centre)."""
    return [{"id": i, "img_name": f"{i:05d}", "width": c.width, "height": c.height,
             "position": np.asarray(c.c2w)[:3, 3].tolist(), "rotation": np.asarray(c.c2w)[:3, :3].tolist(),
             "fx": c.fx, "fy": c.fy} for i, c in enumerate(cams)]


def write_splatv(s4d: Scene4D, path: Path, budget: int | None = 4_000_000) -> Path:
    """splaTV ``.splatv``: uint32 magic 0x674B, uint32 JSON length, JSON chunk list, then one
    RGBA32UI texture of 16 uint32 per splat (xyz float32; quaternion, scale, 9 motion and 4 omega
    coefficients and the temporal centre/scale as half floats; rgba8). The background comes first.
    ``budget``: most splats written (the browser holds them all in GPU memory); the background keeps up
    to 40 % of it and the moving part the rest, the least important splats left out."""
    bg, dy = s4d.background, s4d.dynamic
    nb, nd = len(bg), len(dy)
    if budget and nb + nd > budget:
        nb_keep = min(nb, max(int(budget * 0.4), budget - nd))
        bi = _importance_subset(nb, np.prod(bg.scales, 1) * bg.opacity, nb_keep)
        di = _importance_subset(nd, np.prod(dy.scales, 1) * dy.opacity, budget - len(bi))
    else:
        bi, di = np.arange(nb), np.arange(nd)
    n = len(bi) + len(di)
    means = np.concatenate([bg.means[bi], dy.means[di]]).astype("<f4")
    quats = np.concatenate([bg.quats[bi], dy.quats[di]]).astype(np.float32)
    scales = np.concatenate([bg.scales[bi], dy.scales[di]]).astype(np.float32)
    rgba = np.concatenate([np.concatenate([bg.colors[bi], bg.opacity[bi, None]], 1),
                           np.concatenate([dy.colors[di], dy.opacity[di, None]], 1)])
    motion = np.zeros((n, 9), np.float32)
    omega = np.zeros((n, 4), np.float32)
    if dy.motion is not None:
        motion[len(bi):] = dy.motion[di]
    if dy.omega is not None:
        omega[len(bi):] = dy.omega[di]
    tc = np.concatenate([np.full(len(bi), 0.5, np.float32), dy.t_center[di]])
    ts = np.concatenate([np.full(len(bi), STATIC_T_SCALE, np.float32), dy.t_scale[di]])
    texheight = max(1, math.ceil(4 * n / TEX_WIDTH))
    tex = np.zeros((texheight * TEX_WIDTH // 4, 16), "<u4")
    tex[n:, 15] = _half2(np.full(len(tex) - n, 1e4, np.float32), np.ones(len(tex) - n, np.float32))  # padding
    tex[:n, 0:3] = means.view("<u4")
    tex[:n, 3] = _half2(quats[:, 0], quats[:, 1])
    tex[:n, 4] = _half2(quats[:, 2], quats[:, 3])
    tex[:n, 5] = _half2(scales[:, 0], scales[:, 1])
    tex[:n, 6] = _half2(scales[:, 2], np.zeros(n, np.float32))
    tex[:n, 7] = np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8).view("<u4")[:, 0]
    for k in range(4):
        tex[:n, 8 + k] = _half2(motion[:, 2 * k], motion[:, 2 * k + 1])
    tex[:n, 12] = _half2(motion[:, 8], np.zeros(n, np.float32))
    tex[:n, 13] = _half2(omega[:, 0], omega[:, 1])
    tex[:n, 14] = _half2(omega[:, 2], omega[:, 3])
    tex[:n, 15] = _half2(tc, ts)
    chunk = {"type": "splat", "size": int(tex.nbytes), "texwidth": TEX_WIDTH, "texheight": texheight,
             "cameras": splatv_cameras(s4d.cameras[:1] or s4d.background.cameras[:1]),
             # 2D2VR180 additions (ignored by other players)
             "count": int(n), "background_count": int(len(bi)), "fps": s4d.fps, "frames": s4d.frames,
             "duration_s": round(s4d.duration_s, 4), "playback": s4d.meta.get("playback", "frames"),
             "convention": "opencv", "metric_scale": bool(bg.metric_scale),
             "frame_cameras": [{"c2w": np.asarray(c.c2w).round(6).tolist()} for c in s4d.cameras]}
    js = json.dumps([chunk]).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array([SPLATV_MAGIC, len(js)], "<u4").tobytes())
        f.write(js)
        f.write(tex.tobytes())
    return path


def read_splatv(path: Path) -> Scene4D:
    raw = Path(path).read_bytes()
    magic, jlen = np.frombuffer(raw[:8], "<u4")
    if magic != SPLATV_MAGIC:
        raise SceneError(f"{Path(path).name} is not a .splatv file")
    chunks = json.loads(raw[8:8 + jlen].decode("utf-8"))
    ch = next(c for c in chunks if c.get("type") == "splat")
    tex = np.frombuffer(raw[8 + jlen:8 + jlen + ch["size"]], "<u4").reshape(-1, 16)
    n = int(ch.get("count", len(tex)))
    tex = tex[:n]
    means = np.ascontiguousarray(tex[:, 0:3]).view("<f4").astype(np.float32)
    q0, q1 = _unhalf2(tex[:, 3])
    q2, q3 = _unhalf2(tex[:, 4])
    quats = np.stack([q0, q1, q2, q3], 1)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    s0, s1 = _unhalf2(tex[:, 5])
    s2, _ = _unhalf2(tex[:, 6])
    rgba = np.ascontiguousarray(tex[:, 7]).view(np.uint8).reshape(-1, 4).astype(np.float32) / 255
    motion = np.stack([c for k in range(5) for c in _unhalf2(tex[:, 8 + k])], 1)[:, :9]
    omega = np.stack([*_unhalf2(tex[:, 13]), *_unhalf2(tex[:, 14])], 1)
    tc, ts = _unhalf2(tex[:, 15])
    if "background_count" in ch:          # written by 2D2VR180: the background comes first
        static = np.arange(n) < int(ch["background_count"])
    else:                                 # other files: still splats visible all the time
        static = (ts >= 10.0) & ~np.any(motion != 0, 1) & ~np.any(omega != 0, 1)
    scales = np.stack([s0, s1, s2], 1)
    frames = int(ch.get("frames", 1))
    cams = []
    for c in ch.get("cameras", []):
        c2w = np.eye(4)
        c2w[:3, :3] = np.asarray(c["rotation"])
        c2w[:3, 3] = np.asarray(c["position"])
        cams.append(Camera(int(c["width"]), int(c["height"]), float(c["fx"]), float(c["fy"]), c["width"] / 2,
                           c["height"] / 2, c2w))
    bg = GaussianScene(means[static], scales[static], quats[static], rgba[static, 3], rgba[static, :3],
                       cameras=cams[:1], metric_scale=bool(ch.get("metric_scale", False)))
    d = ~static
    moving = bool(np.any(motion[d] != 0))
    rotating = bool(np.any(omega[d] != 0))
    dy = DynamicSplats(means[d], scales[d], quats[d], rgba[d, 3], rgba[d, :3], tc[d], ts[d],
                       motion[d] if moving else None, omega[d] if rotating else None)
    frame_cams = [Camera(cams[0].width, cams[0].height, cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy,
                         np.asarray(fc["c2w"])) for fc in ch.get("frame_cameras", [])] if cams else []
    return Scene4D(bg, dy, float(ch.get("fps", 12.0)), frames, frame_cams or cams,
                   {"playback": ch.get("playback", "continuous")})
