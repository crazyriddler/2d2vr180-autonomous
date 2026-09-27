"""VR180 / stereo rendering from a 3D scene (never by 2D pixel shifting).

Pipeline: scene → left/right virtual cameras (parallel, separated by the
eye distance) → per-eye render → half-equirectangular projection → SBS/TB
composition → optional FFmpeg encode. Everything the scene does not cover is
left black and reported as unknown coverage; the report states the real
content field of view so a narrow-FOV result is never sold as full VR180.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .render import RenderResult, fill_disocclusions, render_equirect180, render_pinhole
from .scene import Camera, GaussianScene

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class StereoOptions:
    layout: str = "sbs"                 # "sbs" | "tb"
    projection: str = "equirect180"     # "equirect180" | "flat"
    eye_resolution: int = 2048          # per-eye height (equirect: square)
    eye_separation_m: float = 0.064
    head_offset_m: float = 0.0          # move head back (+) from the source camera
    convergence_m: float | None = None  # flat mode: zero-parallax distance (default: median depth)
    crack_fill: bool = True
    fill_holes: bool = True             # background-fill disocclusions (reported as interpolated)

    def validate(self) -> None:
        if self.layout not in ("sbs", "tb"):
            raise ValueError(f"layout must be 'sbs' or 'tb', not {self.layout!r}")
        if self.projection not in ("equirect180", "flat"):
            raise ValueError(f"projection must be 'equirect180' or 'flat', not {self.projection!r}")
        if not (0.0 <= self.eye_separation_m <= 1.0):
            raise ValueError("eye separation must be between 0 and 1 m")
        if self.eye_resolution < 64 or self.eye_resolution > 8192:
            raise ValueError("eye resolution must be between 64 and 8192")


@dataclass
class StereoFrame:
    image: np.ndarray
    left: RenderResult
    right: RenderResult
    metadata: dict = field(default_factory=dict)


def eye_poses(head_c2w: np.ndarray, ipd: float) -> tuple[np.ndarray, np.ndarray]:
    """Parallel eyes: left at -x, right at +x of the head (OpenCV frame)."""
    left = np.array(head_c2w, dtype=np.float64, copy=True)
    right = np.array(head_c2w, dtype=np.float64, copy=True)
    x_axis = head_c2w[:3, 0]
    left[:3, 3] = head_c2w[:3, 3] - x_axis * ipd / 2
    right[:3, 3] = head_c2w[:3, 3] + x_axis * ipd / 2
    return left, right


def content_fov(scene: GaussianScene, head_c2w: np.ndarray) -> dict:
    """Angular extent (degrees) actually covered by scene content."""
    p = (scene.means.astype(np.float64) - head_c2w[:3, 3]) @ head_c2w[:3, :3]
    p = p[p[:, 2] > 0]
    if len(p) == 0:
        return {"horizontal_deg": 0.0, "vertical_deg": 0.0}
    lon = np.degrees(np.arctan2(p[:, 0], p[:, 2]))
    lat = np.degrees(np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2])))
    lo_x, hi_x = np.percentile(lon, [0.5, 99.5])
    lo_y, hi_y = np.percentile(lat, [0.5, 99.5])
    return {"horizontal_deg": round(float(hi_x - lo_x), 2), "vertical_deg": round(float(hi_y - lo_y), 2)}


def compose(left: np.ndarray, right: np.ndarray, layout: str) -> np.ndarray:
    """Left eye is always first: left half (SBS) or top half (TB)."""
    return np.concatenate([left, right], axis=1 if layout == "sbs" else 0)


@dataclass
class StereoPlan:
    """Everything needed to render one stereo frame, independent of renderer."""
    head: np.ndarray
    eyes: tuple[np.ndarray, np.ndarray]
    width: int
    height: int
    meta: dict
    flat: dict | None = None  # {"fx","fy","cx_left","cx_right","cy"} for flat projection


def plan_stereo(scene: GaussianScene, opts: StereoOptions, head_c2w: np.ndarray | None = None,
                ref_camera: Camera | None = None) -> StereoPlan:
    opts.validate()
    head = np.eye(4) if head_c2w is None else np.asarray(head_c2w, np.float64).copy()
    if opts.head_offset_m:
        head[:3, 3] = head[:3, 3] - head[:3, 2] * opts.head_offset_m
    eyes = eye_poses(head, opts.eye_separation_m)
    H = opts.eye_resolution
    meta: dict = {"layout": opts.layout, "projection": opts.projection, "eye_order": "left_first",
                  "eye_separation_m": opts.eye_separation_m, "head_offset_m": opts.head_offset_m,
                  "metric_scale": scene.metric_scale}
    if opts.projection == "equirect180":
        meta["output_fov_deg"] = {"horizontal": 180.0, "vertical": 180.0}
        return StereoPlan(head, eyes, H, H, meta)
    cam = ref_camera or (scene.cameras[0] if scene.cameras else None)
    if cam is None:
        raise ValueError("flat stereo needs a reference camera (intrinsics)")
    s = H / cam.height
    W = int(round(cam.width * s / 2)) * 2
    fx, fy, cx, cy = cam.fx * s, cam.fy * s, cam.cx * s, cam.cy * s
    conv = opts.convergence_m
    if conv is None:
        z = ((scene.means.astype(np.float64) - head[:3, 3]) @ head[:3, :3])[:, 2]
        conv = float(np.median(z[z > 0])) if (z > 0).any() else 2.0
    shift = fx * opts.eye_separation_m / (2 * conv)
    meta["output_fov_deg"] = {"horizontal": round(float(np.degrees(2 * np.arctan(W / (2 * fx)))), 2),
                              "vertical": round(float(np.degrees(2 * np.arctan(H / (2 * fy)))), 2)}
    meta["convergence_m"] = conv
    return StereoPlan(head, eyes, W, H, meta,
                      {"fx": fx, "fy": fy, "cx_left": cx - shift, "cx_right": cx + shift, "cy": cy})


def render_eyes_cpu(scene: GaussianScene, plan: StereoPlan, opts: StereoOptions) -> tuple[RenderResult, RenderResult]:
    lc2w, rc2w = plan.eyes
    if plan.flat is None:
        return (render_equirect180(scene, lc2w, plan.width, plan.height, crack_fill=opts.crack_fill),
                render_equirect180(scene, rc2w, plan.width, plan.height, crack_fill=opts.crack_fill))
    f = plan.flat
    return (render_pinhole(scene, lc2w, plan.width, plan.height, f["fx"], f["fy"], f["cx_left"], f["cy"],
                           crack_fill=opts.crack_fill),
            render_pinhole(scene, rc2w, plan.width, plan.height, f["fx"], f["fy"], f["cx_right"], f["cy"],
                           crack_fill=opts.crack_fill))


def finalize_stereo(scene: GaussianScene, opts: StereoOptions, plan: StereoPlan, left: RenderResult,
                    right: RenderResult, renderer: str = "cpu-reference") -> StereoFrame:
    meta = dict(plan.meta)
    if opts.fill_holes:
        left, right = fill_disocclusions(left), fill_disocclusions(right)
    meta["renderer"] = renderer
    meta["hole_filling"] = "background propagation (interpolated, not observed)" if opts.fill_holes else "off"
    meta["eye_resolution"] = [int(left.rgb.shape[1]), int(left.rgb.shape[0])]
    meta["content_fov_deg"] = content_fov(scene, plan.head)
    meta["coverage"] = {"left": left.coverage_stats(), "right": right.coverage_stats()}
    unknown = max(meta["coverage"]["left"]["unknown_fraction"], meta["coverage"]["right"]["unknown_fraction"])
    cf = meta["content_fov_deg"]
    meta["is_full_vr180"] = bool(opts.projection == "equirect180" and cf["horizontal_deg"] >= 170
                                 and cf["vertical_deg"] >= 170 and unknown < 0.02)
    if opts.projection == "equirect180" and not meta["is_full_vr180"]:
        meta["honesty_note"] = (f"Scene content covers only ~{cf['horizontal_deg']:.0f}°x{cf['vertical_deg']:.0f}° "
                                f"of the 180°x180° VR180 frame; {unknown:.0%} of each eye is unknown (black). "
                                "This is a constrained-FOV stereo view inside a VR180 container, not a "
                                "full hemispherical reconstruction.")
    img = compose(left.rgb, right.rgb, opts.layout)
    return StereoFrame(img, left, right, meta)


def render_stereo(scene: GaussianScene, opts: StereoOptions, head_c2w: np.ndarray | None = None,
                  ref_camera: Camera | None = None) -> StereoFrame:
    plan = plan_stereo(scene, opts, head_c2w, ref_camera)
    left, right = render_eyes_cpu(scene, plan, opts)
    return finalize_stereo(scene, opts, plan, left, right)


# ------------------------------------------------------------------ GPU views
def gpu_views(plan: StereoPlan) -> list[dict]:
    """Pinhole views a GPU splat renderer must produce for this frame:
    5 cube faces per eye (VR180) or one view per eye (flat)."""
    from .render import cube_face_views

    if plan.flat is None:
        face = max(64, plan.height // 2)
        return [v for eye in plan.eyes for v in cube_face_views(eye, face)]
    f = plan.flat
    return [{"c2w": plan.eyes[0], "width": plan.width, "height": plan.height, "fx": f["fx"], "fy": f["fy"],
             "cx": f["cx_left"], "cy": f["cy"]},
            {"c2w": plan.eyes[1], "width": plan.width, "height": plan.height, "fx": f["fx"], "fy": f["fy"],
             "cx": f["cx_right"], "cy": f["cy"]}]


def assemble_gpu(plan: StereoPlan, rendered: list[dict]) -> tuple[RenderResult, RenderResult]:
    """rendered: dicts with rgb (premultiplied uint8), alpha, depth, in gpu_views() order."""
    from .render import cubemap_to_equirect180

    def flat_result(r):
        a = r["alpha"].astype(np.float32)
        cov = a >= 0.5
        rgb = np.zeros_like(r["rgb"])
        col = r["rgb"].astype(np.float32) / np.maximum(a[..., None], 1e-3)
        rgb[cov] = np.clip(col[cov], 0, 255).astype(np.uint8)
        return RenderResult(rgb, cov, np.zeros_like(cov), np.where(cov, 3, 255).astype(np.uint8),
                            np.where(cov, r["depth"].astype(np.float32), np.inf).astype(np.float32))

    if plan.flat is not None:
        return flat_result(rendered[0]), flat_result(rendered[1])
    eyes = []
    for e in range(2):
        faces = rendered[5 * e:5 * e + 5]
        eyes.append(cubemap_to_equirect180(np.stack([f["rgb"] for f in faces]),
                                           np.stack([f["alpha"].astype(np.float32) for f in faces]),
                                           np.stack([f["depth"].astype(np.float32) for f in faces]),
                                           plan.height))
    return eyes[0], eyes[1]


def output_suffix(opts: StereoOptions) -> str:
    """File-name tags recognised by common VR players (DeoVR/Skybox/Quest)."""
    if opts.projection == "equirect180":
        return "_180_LR" if opts.layout == "sbs" else "_180_TB"
    return "_3D_SBS" if opts.layout == "sbs" else "_3D_TB"


def save_stereo_image(frame: StereoFrame, opts: StereoOptions, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}{output_suffix(opts)}"
    img_path = out_dir / f"{name}.jpg"
    Image.fromarray(frame.image).save(img_path, quality=95, subsampling=0)
    def cov_img(r):  # 255 observed/inferred, 160 crack-filled, 90 hole-filled, 0 unknown
        m = np.zeros(r.covered.shape, np.uint8)
        if r.hole_filled is not None:
            m[r.hole_filled] = 90
        m[r.filled] = 160
        m[r.covered] = 255
        return m

    mask = np.concatenate([cov_img(frame.left), cov_img(frame.right)], axis=1 if opts.layout == "sbs" else 0)
    mask_path = out_dir / f"{name}_coverage.png"
    Image.fromarray(mask).save(mask_path)
    meta = dict(frame.metadata)
    meta.update({"image": img_path.name, "coverage_mask": mask_path.name,
                 "width": int(frame.image.shape[1]), "height": int(frame.image.shape[0])})
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return {"image": str(img_path), "coverage_mask": str(mask_path), "metadata": meta}


def encode_video(frames_iter, width: int, height: int, fps: float, out_path: Path, opts: StereoOptions,
                 ffmpeg: str, crf: int = 18) -> dict:
    """Encode RGB frames to H.264 MP4 (yuv420p, Quest-compatible)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stereo_mode = "left_right" if opts.layout == "sbs" else "top_bottom"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-metadata:s:v:0", f"stereo_mode={stereo_mode}", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    n = 0
    try:
        assert proc.stdin is not None
        for fr in frames_iter:
            if fr.shape != (height, width, 3):
                raise ValueError(f"frame shape {fr.shape} != {(height, width, 3)}")
            proc.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
            n += 1
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"FFmpeg encode failed ({rc}): {err[-500:]}")
    from .spatial import SpatialError, inject

    try:
        spherical = inject(out_path, opts.layout,
                           "equirect180" if opts.projection == "equirect180" else "none")
    except SpatialError as e:
        spherical = {"error": str(e), "fallback": "filename tag only"}
    return {"path": str(out_path), "frames": n, "fps": fps, "codec": "h264", "pix_fmt": "yuv420p",
            "width": width, "height": height, "stereo_mode": stereo_mode, "spherical_metadata": spherical}


def still_to_video(frame: StereoFrame, opts: StereoOptions, out_dir: Path, stem: str, ffmpeg: str,
                   seconds: float = 5.0, fps: float = 30.0) -> dict:
    h, w = frame.image.shape[:2]
    nframes = max(1, int(seconds * fps))
    return encode_video((frame.image for _ in range(nframes)), w, h, fps,
                        out_dir / f"{stem}{output_suffix(opts)}.mp4", opts, ffmpeg)


def options_dict(opts: StereoOptions) -> dict:
    return asdict(opts)
