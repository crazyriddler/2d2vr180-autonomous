"""Unified scene model: Gaussian splats with per-splat provenance.

Coordinate convention everywhere inside the application: OpenCV camera frame
of the *reference* (source) camera — x right, y down, z forward, metres when
the backend provides metric scale. Backends with other conventions convert in
their adapter.

Provenance codes (never conflate observed and generated geometry):
    0 OBSERVED   — reconstructed from multiple observations (multi-view)
    1 INFERRED   — seen in the input but depth/shape is predicted (monocular)
    2 GENERATIVE — hallucinated by a generative completion model
    3 UNKNOWN    — backend could not say
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

OBSERVED, INFERRED, GENERATIVE, UNKNOWN = 0, 1, 2, 3
PROVENANCE_NAMES = {OBSERVED: "observed", INFERRED: "inferred", GENERATIVE: "generative", UNKNOWN: "unknown"}


class SceneError(Exception):
    pass


@dataclass
class Camera:
    """Pinhole camera. ``c2w`` maps camera (OpenCV) to scene coordinates."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.height / (2 * self.fy))))

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy, "c2w": np.asarray(self.c2w).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]),
                   float(d["cx"]), float(d["cy"]), np.asarray(d.get("c2w", np.eye(4)), dtype=np.float64))


@dataclass
class GaussianScene:
    means: np.ndarray              # (N,3) float32
    scales: np.ndarray             # (N,3) float32, linear (not log)
    quats: np.ndarray              # (N,4) float32, (w,x,y,z), normalised
    opacity: np.ndarray            # (N,)  float32 in [0,1]
    colors: np.ndarray             # (N,3) float32 in [0,1] (display/sRGB-like, DC term)
    provenance: np.ndarray | None = None   # (N,) uint8
    sh_rest: np.ndarray | None = None      # (N,K) float32 higher-order SH, preserved verbatim
    cameras: list[Camera] = field(default_factory=list)
    metric_scale: bool = False

    def __post_init__(self) -> None:
        n = len(self.means)
        if self.provenance is None:
            self.provenance = np.full(n, UNKNOWN, np.uint8)
        for name, arr, shape in (("means", self.means, (n, 3)), ("scales", self.scales, (n, 3)),
                                 ("quats", self.quats, (n, 4)), ("opacity", self.opacity, (n,)),
                                 ("colors", self.colors, (n, 3)), ("provenance", self.provenance, (n,))):
            if arr.shape != shape:
                raise SceneError(f"{name} has shape {arr.shape}, expected {shape}")

    def __len__(self) -> int:
        return len(self.means)

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Return a list of problems; empty when the scene is sane."""
        problems = []
        if len(self) == 0:
            problems.append("scene is empty")
            return problems
        for name in ("means", "scales", "quats", "opacity", "colors"):
            arr = getattr(self, name)
            bad = ~np.isfinite(arr)
            if bad.any():
                problems.append(f"{name} contains {int(bad.sum())} NaN/Inf values")
        if (self.scales <= 0).any():
            problems.append("non-positive scales")
        ext = np.percentile(np.abs(self.means[np.isfinite(self.means).all(1)]), 99) if len(self) else 0
        if ext > 1e5:
            problems.append(f"geometry explosion: 99th percentile |coord| = {ext:.3g}")
        return problems

    def finite_subset(self) -> "GaussianScene":
        ok = (np.isfinite(self.means).all(1) & np.isfinite(self.scales).all(1) & np.isfinite(self.quats).all(1)
              & np.isfinite(self.opacity) & np.isfinite(self.colors).all(1) & (self.scales > 0).all(1))
        return self.subset(ok)

    def subset(self, mask: np.ndarray) -> "GaussianScene":
        return GaussianScene(self.means[mask], self.scales[mask], self.quats[mask], self.opacity[mask],
                             self.colors[mask], self.provenance[mask],
                             None if self.sh_rest is None else self.sh_rest[mask],
                             list(self.cameras), self.metric_scale)

    def coverage(self) -> dict:
        n = max(len(self), 1)
        counts = np.bincount(self.provenance, minlength=4)
        return {PROVENANCE_NAMES[k]: round(float(counts[k]) / n, 6) for k in range(4)}

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.percentile(self.means, 1, axis=0)
        hi = np.percentile(self.means, 99, axis=0)
        return lo, hi


# ------------------------------------------------------------------ PLY I/O

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_elements(path: Path) -> dict[str, np.ndarray]:
    """Minimal PLY reader: binary (LE/BE) and ASCII, scalar properties only
    for elements preceding any list-property element."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise SceneError(f"{path.name} is not a PLY file")
        fmt = None
        elements: list[tuple[str, int, list[tuple[str, str]], bool]] = []
        while True:
            line = f.readline()
            if not line:
                raise SceneError("truncated PLY header")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elements.append((tok[1], int(tok[2]), [], False))
            elif tok[0] == "property":
                name, cnt, props, has_list = elements[-1]
                if tok[1] == "list":
                    elements[-1] = (name, cnt, props, True)
                else:
                    props.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        out: dict[str, np.ndarray] = {}
        if fmt == "ascii":
            name, cnt, props, has_list = elements[0]
            if has_list:
                raise SceneError("ASCII PLY with list properties is not supported")
            data = np.loadtxt(f, max_rows=cnt, ndmin=2)
            out[name] = np.rec.fromarrays(data.T, names=[p[0] for p in props])
            return out
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise SceneError(f"unsupported PLY format {fmt}")
        endian = "<" if fmt == "binary_little_endian" else ">"
        for name, cnt, props, has_list in elements:
            if has_list:
                break  # e.g. mesh faces; not needed for splats
            dt = np.dtype([(p, endian + t) for p, t in props])
            buf = f.read(dt.itemsize * cnt)
            if len(buf) < dt.itemsize * cnt:
                raise SceneError(f"PLY element '{name}' truncated")
            out[name] = np.frombuffer(buf, dt, count=cnt)
        return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_gaussian_ply(path: Path) -> GaussianScene:
    els = read_ply_elements(Path(path))
    if "vertex" not in els:
        raise SceneError("PLY has no vertex element")
    v = els["vertex"]
    names = v.dtype.names or ()
    need = ("x", "y", "z")
    if not all(k in names for k in need):
        raise SceneError("PLY vertices lack x/y/z")
    n = len(v)
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    if "f_dc_0" in names:
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc, 0, 1)
    elif "red" in names:
        colors = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)
        colors = colors / (255.0 if v["red"].dtype.kind == "u" else 1.0)
    else:
        colors = np.full((n, 3), 0.7, np.float32)
    rest_names = sorted([k for k in names if k.startswith("f_rest_")], key=lambda s: int(s[7:]))
    sh_rest = np.stack([v[k] for k in rest_names], 1).astype(np.float32) if rest_names else None
    if "opacity" in names:
        opacity = _sigmoid(v["opacity"].astype(np.float32))
    else:
        opacity = np.ones(n, np.float32)
    if "scale_0" in names:
        scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32))
    else:  # plain point cloud: estimate a footprint from density
        lo, hi = np.percentile(means, 2, 0), np.percentile(means, 98, 0)
        s = float(np.linalg.norm(hi - lo)) / max(np.cbrt(n), 1) * 0.5
        scales = np.full((n, 3), max(s, 1e-4), np.float32)
    if "rot_0" in names:
        q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    else:
        q = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    prov = v["provenance"].astype(np.uint8) if "provenance" in names else None
    side = _sidecar(path, ".provenance.npy")
    if prov is None and side.exists():
        arr = np.load(side)
        if arr.shape == (n,):
            prov = arr.astype(np.uint8)
    scene = GaussianScene(means, scales.astype(np.float32), q, opacity.astype(np.float32),
                          colors.astype(np.float32), prov, sh_rest)
    if "intrinsic" in els and "image_size" in els:  # SHARP-style metadata
        k = np.asarray(els["intrinsic"]["intrinsic"], np.float64).reshape(3, 3)
        wh = np.asarray(els["image_size"]["image_size"]).astype(int)
        scene.cameras.append(Camera(int(wh[0]), int(wh[1]), k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
    meta = _sidecar(path, ".meta.json")
    if meta.exists():
        m = json.loads(meta.read_text())
        scene.cameras = [Camera.from_dict(c) for c in m.get("cameras", [])] or scene.cameras
        scene.metric_scale = bool(m.get("metric_scale", False))
    return scene


SIDECAR_DIR = "_2d2vr180"  # app metadata kept out of the way of the user-facing files


def _sidecar(ply: Path, suffix: str) -> Path:
    """Sidecar location: <dir>/_2d2vr180/<stem><suffix>; older builds wrote <stem><suffix> next to the PLY."""
    ply = Path(ply)
    new = ply.parent / SIDECAR_DIR / (ply.stem + suffix)
    return new if new.exists() else ply.with_suffix(suffix)


def write_gaussian_ply(scene: GaussianScene, path: Path, include_provenance: bool = False) -> Path:
    """Write the de-facto standard 3DGS PLY (INRIA layout, binary LE) plus a
    ``_2d2vr180/<stem>.meta.json`` sidecar with cameras and provenance summary."""
    path = Path(path)
    n = len(scene)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    k = 0 if scene.sh_rest is None else scene.sh_rest.shape[1]
    props += [f"f_rest_{i}" for i in range(k)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dt = [(p, "<f4") for p in props]
    if include_provenance:
        dt.append(("provenance", "u1"))
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = scene.means.T
    dc = (np.clip(scene.colors, 0, 1) - 0.5) / SH_C0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    for i in range(k):
        arr[f"f_rest_{i}"] = scene.sh_rest[:, i]
    op = np.clip(scene.opacity, 1e-6, 1 - 1e-6)
    arr["opacity"] = np.log(op / (1 - op))
    ls = np.log(np.maximum(scene.scales, 1e-9))
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = ls.T
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = scene.quats.T
    if include_provenance:
        arr["provenance"] = scene.provenance
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for name, t in dt:
        header.append(f"property {'float' if t == '<f4' else 'uchar'} {name}")
    header.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes())
    side_dir = path.parent / SIDECAR_DIR
    side_dir.mkdir(exist_ok=True)
    write_scene_meta(scene, side_dir / (path.stem + ".meta.json"))
    np.save(side_dir / (path.stem + ".provenance.npy"), scene.provenance)
    return path


def write_scene_meta(scene: GaussianScene, path: Path) -> None:
    Path(path).write_text(json.dumps({
        "convention": "opencv_camera (x right, y down, z forward)",
        "metric_scale": scene.metric_scale,
        "count": len(scene),
        "coverage_by_splat": scene.coverage(),
        "cameras": [c.to_dict() for c in scene.cameras],
    }, indent=2))


def write_splat(scene: GaussianScene, path: Path) -> Path:
    """antimatter15 ``.splat`` (32 bytes/splat), sorted by importance."""
    n = len(scene)
    vol = np.prod(scene.scales, axis=1)
    order = np.argsort(-(vol * scene.opacity))
    buf = np.zeros(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["pos"] = scene.means[order]
    buf["scale"] = scene.scales[order]
    rgba = np.concatenate([scene.colors[order], scene.opacity[order, None]], 1)
    buf["rgba"] = np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8)
    q = scene.quats[order] / (np.linalg.norm(scene.quats[order], axis=1, keepdims=True) + 1e-12)
    buf["rot"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    Path(path).write_bytes(buf.tobytes())
    return Path(path)


def read_splat(path: Path) -> GaussianScene:
    raw = Path(path).read_bytes()
    if len(raw) % 32:
        raise SceneError(".splat size is not a multiple of 32 bytes")
    buf = np.frombuffer(raw, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    q = (buf["rot"].astype(np.float32) - 128) / 128
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    rgba = buf["rgba"].astype(np.float32) / 255
    return GaussianScene(buf["pos"].astype(np.float32).copy(), buf["scale"].astype(np.float32).copy(), q,
                         rgba[:, 3].copy(), rgba[:, :3].copy())


def load_scene(path: Path) -> GaussianScene:
    path = Path(path)
    if path.suffix.lower() == ".splat":
        return read_splat(path)
    return read_gaussian_ply(path)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)

