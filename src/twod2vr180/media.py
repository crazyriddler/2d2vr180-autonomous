"""Media ingestion: input classification, FFmpeg I/O, video analysis.

Video analysis exists so the pipeline selector never burns ten minutes on 500
nearly identical frames: a fixed camera looking at a static scene carries no
more geometric information than its sharpest frame.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".wmv", ".mpg", ".mpeg"}

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class MediaError(Exception):
    """Invalid, unsupported or corrupted input media."""


def find_ffmpeg() -> str | None:
    """Bundled FFmpeg first, then the imageio-ffmpeg wheel, then PATH."""
    from .paths import bundled_bin_dir

    override = os.environ.get("TWOD2VR180_FFMPEG")
    if override and Path(override).exists():
        return override
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    cand = bundled_bin_dir() / exe
    if cand.exists():
        return str(cand)
    try:
        import imageio_ffmpeg  # type: ignore

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and Path(path).exists():
            return path
    except Exception:
        pass
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    ff = find_ffmpeg()
    if not ff:
        raise MediaError("FFmpeg is not available; video input/output is disabled.")
    return ff


def classify_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "photo"
    if ext in VIDEO_EXTS:
        return "video"
    raise MediaError(f"Unsupported file type '{ext}'. Supported: images "
                     f"{sorted(IMAGE_EXTS)} and videos {sorted(VIDEO_EXTS)}.")


# ---------------------------------------------------------------- images

def load_image(path: Path, max_side: int | None = None) -> np.ndarray:
    """Load an image as uint8 RGB (H, W, 3), honouring EXIF orientation."""
    from PIL import ImageOps

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            if max_side and max(im.size) > max_side:
                s = max_side / max(im.size)
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8).copy()
    except (OSError, SyntaxError, ValueError) as e:
        raise MediaError(f"Cannot decode image {path.name}: {e}") from e
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        raise MediaError(f"Image {path.name} is too small ({arr.shape[1]}x{arr.shape[0]}).")
    return arr


# ---------------------------------------------------------------- video

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float | None
    codec: str | None
    rotation: int = 0

    @property
    def est_frames(self) -> int | None:
        return int(round(self.duration_s * self.fps)) if self.duration_s else None


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #\S+.*?Video:\s*([^\s,]+).*?,\s*(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*tbr")
_ROT_RE = re.compile(r"(?:rotate\s*:\s*|rotation of\s*)(-?\d+(?:\.\d+)?)")


def parse_ffmpeg_probe(stderr: str) -> VideoInfo:
    m = _VID_RE.search(stderr)
    if not m:
        raise MediaError("No video stream found (file is corrupted or not a video).")
    codec, w, h = m.group(1), int(m.group(2)), int(m.group(3))
    line = stderr[m.start(): stderr.find("\n", m.start())]
    fm = _FPS_RE.search(line) or _TBR_RE.search(line)
    fps = float(fm.group(1)) if fm else 30.0
    dur = None
    dm = _DUR_RE.search(stderr)
    if dm:
        dur = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
    rot = 0
    rm = _ROT_RE.search(stderr)
    if rm:
        rot = int(round(float(rm.group(1)))) % 360
    if rot in (90, 270):  # ffmpeg autorotates on decode
        w, h = h, w
    return VideoInfo(width=w, height=h, fps=fps, duration_s=dur, codec=codec, rotation=rot)


def probe_video(path: Path) -> VideoInfo:
    ff = require_ffmpeg()
    if not path.exists():
        raise MediaError(f"File not found: {path}")
    proc = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                          errors="replace", creationflags=_NO_WINDOW)
    return parse_ffmpeg_probe(proc.stderr)


def iter_video_frames(path: Path, width: int | None = None, fps: float | None = None,
                      gray: bool = False, info: VideoInfo | None = None):
    """Yield decoded frames as uint8 arrays using an FFmpeg rawvideo pipe."""
    ff = require_ffmpeg()
    info = info or probe_video(path)
    if width and width < info.width:
        w = width - width % 2
        h = max(2, int(round(info.height * w / info.width / 2)) * 2)
    else:
        w, h = info.width, info.height
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    vf.append(f"scale={w}:{h}:flags=area")
    pix = "gray" if gray else "rgb24"
    ch = 1 if gray else 3
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", ",".join(vf),
           "-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_NO_WINDOW)
    frame_bytes = w * h * ch
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(h, w) if gray else \
                np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield arr
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def extract_frames(path: Path, indices: list[int], src_fps: float, out_dir: Path,
                   max_side: int | None = None) -> list[Path]:
    """Extract full-resolution frames at the given frame indices as PNG."""
    ff = require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, idx in enumerate(indices):
        t = idx / src_fps
        out = out_dir / f"frame_{n:05d}.png"
        vf = []
        if max_side:
            vf = ["-vf", f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"]
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.4f}", "-i", str(path),
               *vf, "-frames:v", "1", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
        if r.returncode != 0 or not out.exists():
            raise MediaError(f"Failed to extract frame {idx}: {r.stderr.strip()[-300:]}")
        paths.append(out)
    return paths


# ---------------------------------------------------------------- analysis

def sharpness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def phase_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Translation (dy, dx) of b relative to a and the peak strength (0..1)."""
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    fa = np.fft.rfft2(a * win)
    fb = np.fft.rfft2(b * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    corr = np.fft.irfft2(r, s=a.shape)
    idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[idx])
    dy, dx = idx
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(dy), float(dx), peak


@dataclass
class VideoAnalysis:
    info: dict
    sampled_frames: int
    sample_fps: float
    camera_motion: str          # "static" | "moving"
    scene_dynamics: str         # "static" | "dynamic"
    kind: str                   # "static_scene" | "static_camera_dynamic" | "moving_camera"
    median_shift_px: float
    dynamic_fraction: float
    cuts: list[int]             # source frame indices where a new shot starts
    keyframes: list[int]        # source frame indices recommended for reconstruction
    best_frame: int
    redundant_fraction: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_frames(frames: list[np.ndarray], sample_fps: float, src_fps: float,
                   max_keyframes: int = 80, motion_thresh: float = 0.004,
                   dyn_thresh: float = 0.01) -> VideoAnalysis:
    """Classify a sequence of small grayscale frames.

    ``motion_thresh`` is the median inter-sample translation as a fraction of
    frame width above which the camera is considered moving. ``dyn_thresh`` is
    the fraction of pixels with significant temporal change above which a
    static-camera shot is considered to contain moving subjects.
    """
    if not frames:
        raise MediaError("Video contains no decodable frames.")
    n = len(frames)
    h, w = frames[0].shape
    step = src_fps / sample_fps
    src_idx = [min(int(round(i * step)), 10**9) for i in range(n)]
    sharp = [sharpness(f) for f in frames]
    notes: list[str] = []

    shifts, diffs = [], []
    for i in range(1, n):
        dy, dx, _ = phase_correlation(frames[i - 1], frames[i])
        shifts.append(float(np.hypot(dy, dx)))
        diffs.append(float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)))))

    # Scene cuts: a difference far above the running typical level.
    cuts: list[int] = []
    if diffs:
        med = float(np.median(diffs)) + 1.0
        for i, d in enumerate(diffs, start=1):
            if d > max(25.0, 6.0 * med):
                cuts.append(src_idx[i])
    if cuts:
        notes.append(f"{len(cuts)} scene cut(s) detected; segments are reconstructed independently.")

    median_shift = float(np.median(shifts)) if shifts else 0.0
    moving = median_shift > motion_thresh * w

    stack = np.stack(frames).astype(np.float32)
    if n >= 3:
        med_frame = np.median(stack, axis=0)
        changed = np.abs(stack - med_frame) > 18.0
        dyn_frac = float(np.mean(changed.mean(axis=0) > 0.1))
    else:
        dyn_frac = 0.0
    dynamic = dyn_frac > dyn_thresh

    best = src_idx[int(np.argmax(sharp))]
    if moving:
        kind = "moving_camera"
        # Keep a frame whenever accumulated motion exceeds ~2% of the width,
        # always keeping the first; prefer the sharper of neighbours.
        keep = [0]
        acc = 0.0
        for i, s in enumerate(shifts, start=1):
            acc += s
            if acc >= 0.02 * w:
                j = i if sharp[i] >= sharp[i - 1] or (i - 1) in keep else i - 1
                if j not in keep:
                    keep.append(j)
                acc = 0.0
        if len(keep) > max_keyframes:
            sel = np.linspace(0, len(keep) - 1, max_keyframes).round().astype(int)
            keep = [keep[k] for k in sel]
        keyframes = [src_idx[k] for k in keep]
        if len(keyframes) < 3:
            notes.append("Camera moves very little; multi-view reconstruction may be weak.")
    elif dynamic:
        kind = "static_camera_dynamic"
        keyframes = src_idx[:]  # every sample; per-frame processing
        notes.append("Fixed camera with moving subjects: static multi-view reconstruction would be "
                     "invalid; per-frame processing is used and background is shared.")
    else:
        kind = "static_scene"
        keyframes = [best]
        notes.append(f"Fixed camera and static scene: {n - 1} of {n} sampled frames are redundant; "
                     "the sharpest frame is reconstructed as a photo.")
    redundant = 1.0 - len(keyframes) / n
    return VideoAnalysis(info={}, sampled_frames=n, sample_fps=sample_fps,
                         camera_motion="moving" if moving else "static",
                         scene_dynamics="dynamic" if dynamic else "static", kind=kind,
                         median_shift_px=median_shift, dynamic_fraction=dyn_frac, cuts=cuts,
                         keyframes=keyframes, best_frame=best, redundant_fraction=redundant, notes=notes)


def analyze_video(path: Path, max_samples: int = 120, analysis_width: int = 192,
                  max_keyframes: int = 80) -> VideoAnalysis:
    info = probe_video(path)
    dur = info.duration_s or 10.0
    sample_fps = min(info.fps, max(0.5, max_samples / max(dur, 1e-3)))
    frames = []
    for f in iter_video_frames(path, width=analysis_width, fps=sample_fps, gray=True, info=info):
        frames.append(f)
        if len(frames) >= max_samples:
            break
    res = analyze_frames(frames, sample_fps, info.fps, max_keyframes=max_keyframes)
    res.info = asdict(info)
    return res


def extract_frames_fps(path: Path, out_dir: Path, fps: float, max_frames: int,
                       max_side: int | None = None) -> list[Path]:
    """Extract frames at a fixed rate in a single FFmpeg pass."""
    ff = require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps}"
    if max_side:
        vf += f",scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-vf", vf,
           "-frames:v", str(max_frames), str(out_dir / "frame_%05d.png")]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    frames = sorted(out_dir.glob("frame_*.png"))
    if r.returncode != 0 or not frames:
        raise MediaError(f"Frame extraction failed: {r.stderr.strip()[-300:]}")
    return frames


def exif_hfov_deg(path: Path) -> float | None:
    """Horizontal field of view from EXIF 35 mm-equivalent focal length."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            f35 = exif.get_ifd(0x8769).get(0xA405) or exif.get(0xA405)
            w, h = im.size
    except Exception:
        return None
    if not f35:
        return None
    # 35 mm equivalent refers to the 36 mm-wide frame along the long side.
    hfov_long = 2 * np.degrees(np.arctan(36.0 / (2 * float(f35))))
    if w >= h:
        return float(hfov_long)
    return float(2 * np.degrees(np.arctan(np.tan(np.radians(hfov_long) / 2) * w / h)))
