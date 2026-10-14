"""SPLV: a splat video in one file (2D2VR 2.0). numpy + zlib only, shared by the video worker (writes frames)
and the app (assembles the file, reads it for the viewer). The browser player (assets/webxr/player.js)
decodes the same layout.

Every frame holds the same K splats in the same order (a fixed subset of SHARP's pixel-aligned grid), so a
splat that does not move keeps its quantized values from one frame to the next. Frames are stored as
differences to the previous frame (keyframes every ``keyframe_interval`` frames are stored whole), which
compresses still parts of a video to almost nothing.

File:
  b"SPLV" | u32 version (1) | u32 header length | header (JSON, utf-8) | zero padding to 8 bytes | data
Header (JSON):
  version, fps, frames, count (K), keyframe_interval, quant {u, v, logz, logs: [lo, hi]},
  camera {width, height, fx, fy, cx, cy} (OpenCV, the filming camera; splats are in its frame),
  index [[offset, size, keyframe], ...] (offsets from the start of the data section),
  audio {offset, size, mime} | null, plus free-form metadata (app, source, scale).
Frame = 14 filter bytes (one per field) + a zlib/deflate stream of planar arrays of K values each:
  u, v, z: uint16  (u = x/z, v = y/z mapped linearly to [0, 65535]; z = log depth mapped likewise)
  s0, s1, s2: uint8 (log(scale / z), per axis)
  q0..q3: uint8    (rotation quaternion w, x, y, z with w >= 0, mapped from [-1, 1])
  r, g, b, a: uint8 (colour, opacity)
Filters (bit 0: temporal, bit 1: spatial), applied by the encoder in this order and undone in reverse:
  temporal: value - the same splat's value in the previous frame (never in a keyframe);
  spatial:  value - the previous splat's value (splats are in image order, so neighbours are alike).
All arithmetic is modulo 2^16 / 2^8. A uint16 array is stored as two byte planes (low bytes, then high
bytes), which deflate compresses much better. The encoder picks each field's filters per frame.
"""

import json
import struct
import zlib

import numpy as np

MAGIC = b"SPLV"
VERSION = 1
FIELDS = (("u", np.uint16), ("v", np.uint16), ("z", np.uint16), ("s0", np.uint8), ("s1", np.uint8),
          ("s2", np.uint8), ("q0", np.uint8), ("q1", np.uint8), ("q2", np.uint8), ("q3", np.uint8),
          ("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8))
BYTES_PER_SPLAT = sum(np.dtype(t).itemsize for _, t in FIELDS)     # 17


def default_quant(hfov_deg: float = 90.0, vfov_deg: float = 70.0) -> dict:
    """Quantization ranges: the image plane with a margin (splats can stick out of the frame), depths
    5 cm - 300 m, splat sizes from 1e-6 to 0.37 of their distance."""
    import math

    tu = math.tan(math.radians(min(hfov_deg, 170) / 2)) * 2.5
    tv = math.tan(math.radians(min(vfov_deg, 170) / 2)) * 2.5
    return {"u": [-tu, tu], "v": [-tv, tv], "logz": [math.log(0.05), math.log(300.0)], "logs": [-14.0, -1.0]}


def _q(x, lo, hi, levels):
    return np.clip(np.rint((np.asarray(x, np.float64) - lo) / (hi - lo) * levels), 0, levels)


def quantize(means, scales, quats, colors, opacity, quant: dict) -> dict:
    """Splats (camera frame, metres; quats w,x,y,z; colours and opacity in [0, 1]) → planar arrays."""
    z = np.maximum(np.asarray(means[:, 2], np.float64), 1e-4)
    out = {"u": _q(means[:, 0] / z, *quant["u"], 65535).astype(np.uint16),
           "v": _q(means[:, 1] / z, *quant["v"], 65535).astype(np.uint16),
           "z": _q(np.log(z), *quant["logz"], 65535).astype(np.uint16)}
    ls = np.log(np.maximum(np.asarray(scales, np.float64), 1e-12) / z[:, None])
    for i in range(3):
        out[f"s{i}"] = _q(ls[:, i], *quant["logs"], 255).astype(np.uint8)
    q = np.asarray(quats, np.float64)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    q = np.where(q[:, :1] < 0, -q, q)
    for i in range(4):
        out[f"q{i}"] = _q(q[:, i], -1.0, 1.0, 255).astype(np.uint8)
    c = np.clip(np.asarray(colors, np.float64), 0, 1)
    for i, k in enumerate("rgb"):
        out[k] = np.rint(c[:, i] * 255).astype(np.uint8)
    out["a"] = np.rint(np.clip(np.asarray(opacity, np.float64), 0, 1) * 255).astype(np.uint8)
    return out


def dequantize(arr: dict, quant: dict) -> dict:
    """Planar arrays → means, scales, quats (w,x,y,z), colors, opacity (float32)."""
    def lin(a, lo, hi, levels):
        return lo + a.astype(np.float64) / levels * (hi - lo)

    z = np.exp(lin(arr["z"], *quant["logz"], 65535))
    means = np.stack([lin(arr["u"], *quant["u"], 65535) * z, lin(arr["v"], *quant["v"], 65535) * z, z], 1)
    scales = np.stack([np.exp(lin(arr[f"s{i}"], *quant["logs"], 255)) * z for i in range(3)], 1)
    quats = np.stack([lin(arr[f"q{i}"], -1.0, 1.0, 255) for i in range(4)], 1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    colors = np.stack([arr[k].astype(np.float64) / 255 for k in "rgb"], 1)
    return {"means": means.astype(np.float32), "scales": scales.astype(np.float32),
            "quats": quats.astype(np.float32), "colors": colors.astype(np.float32),
            "opacity": (arr["a"].astype(np.float32) / 255)}


TEMPORAL, SPATIAL = 1, 2


def _spatial(a):
    d = a.copy()
    d[1:] = a[1:] - a[:-1]                     # unsigned arithmetic wraps: modulo 2^n
    return d


def _cost(a):
    """Rough size of an array after deflate: the mean magnitude of its values read as signed."""
    signed = a.view(np.int16 if a.dtype == np.uint16 else np.int8)
    return float(np.abs(signed.astype(np.int32)).mean())


def _planes(a) -> bytes:
    if a.dtype == np.uint16:
        b = a.astype("<u2").view(np.uint8).reshape(-1, 2)
        return np.ascontiguousarray(b[:, 0]).tobytes() + np.ascontiguousarray(b[:, 1]).tobytes()
    return np.ascontiguousarray(a).tobytes()


class FastDequantizer:
    """Lookup-table dequantization in float32 for playback (the app's viewer): means, scales and colours of a
    frame several times faster than :func:`dequantize`."""

    def __init__(self, quant: dict):
        def lut(lo, hi, levels, f=None):
            x = lo + np.arange(levels + 1, dtype=np.float64) / levels * (hi - lo)
            return (f(x) if f else x).astype(np.float32)

        self.u, self.v = lut(*quant["u"], 65535), lut(*quant["v"], 65535)
        self.z, self.s = lut(*quant["logz"], 65535, np.exp), lut(*quant["logs"], 255, np.exp)
        self.c = (np.arange(256, dtype=np.float32) / 255)

    def __call__(self, arr: dict) -> dict:
        z = self.z[arr["z"]]
        means = np.empty((len(z), 3), np.float32)
        np.multiply(self.u[arr["u"]], z, out=means[:, 0])
        np.multiply(self.v[arr["v"]], z, out=means[:, 1])
        means[:, 2] = z
        scales = np.empty_like(means)
        for i in range(3):
            np.multiply(self.s[arr[f"s{i}"]], z, out=scales[:, i])
        colors = np.empty_like(means)
        for i, k in enumerate("rgb"):
            colors[:, i] = self.c[arr[k]]
        return {"means": means, "scales": scales, "colors": colors, "opacity": self.c[arr["a"]]}

    def packed(self, arr: dict) -> np.ndarray:
        """(K, 7) float32 rows of x, y, z, r, g, b and the footprint radius (1.5 sigma of the two largest axes):
        the app viewer's vertex layout, made from 1-D lookups only."""
        z = self.z[arr["z"]]
        out = np.empty((len(z), 7), np.float32)
        np.multiply(self.u[arr["u"]], z, out=out[:, 0])
        np.multiply(self.v[arr["v"]], z, out=out[:, 1])
        out[:, 2] = z
        for i, k in enumerate("rgb"):
            out[:, 3 + i] = self.c[arr[k]]
        if not hasattr(self, "s2"):
            self.s2 = self.s.astype(np.float64) ** 2
        a, b, c = self.s2[arr["s0"]], self.s2[arr["s1"]], self.s2[arr["s2"]]
        m = np.minimum(np.minimum(a, b), c)
        t = a + b + c - m
        np.sqrt(t * 0.5, out=t)
        np.multiply(t, 1.5 * z, out=t)
        out[:, 6] = t
        return out


def encode_frame(arr: dict, prev: dict | None) -> bytes:
    """One frame: whole (prev None, a keyframe) or using ``prev``; each field with its best filters."""
    modes, parts = bytearray(), []
    for name, t in FIELDS:
        a = np.ascontiguousarray(arr[name], t)
        cands = [(0, a), (SPATIAL, _spatial(a))]
        if prev is not None:
            ta = a - np.asarray(prev[name], t)
            cands += [(TEMPORAL, ta), (TEMPORAL | SPATIAL, _spatial(ta))]
        mode, best = min(cands, key=lambda c: _cost(c[1]))
        modes.append(mode)
        parts.append(_planes(best))
    return bytes(modes) + zlib.compress(b"".join(parts), 6)


def decode_frame(data: bytes, count: int, prev: dict | None) -> dict:
    n = len(FIELDS)
    modes, raw = data[:n], zlib.decompress(data[n:])
    out, o = {}, 0
    for (name, t), mode in zip(FIELDS, modes):
        if t == np.uint16:
            lo = np.frombuffer(raw, np.uint8, count, o)
            hi = np.frombuffer(raw, np.uint8, count, o + count)
            a = lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)
            o += 2 * count
        else:
            a = np.frombuffer(raw, np.uint8, count, o).copy()
            o += count
        if mode & SPATIAL:
            a = np.cumsum(a, dtype=t)
        if mode & TEMPORAL:
            if prev is None:
                raise ValueError("a frame that needs the previous frame was decoded without it")
            a = a + np.asarray(prev[name], t)
        out[name] = a
    return out


def write_file(path, header: dict, chunks, audio: bytes | None = None, audio_mime: str = "audio/mp4") -> dict:
    """Assemble a .splv file from frame chunks (an iterable of (bytes, is_key)) and optional audio. The
    chunks are streamed: memory stays at one frame. Returns the header written."""
    import os

    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    index, offset = [], 0
    tmp = path + ".data"
    with open(tmp, "wb") as f:
        for data, key in chunks:
            f.write(data)
            index.append([offset, len(data), bool(key)])
            offset += len(data)
        audio_info = None
        if audio:
            f.write(audio)
            audio_info = {"offset": offset, "size": len(audio), "mime": audio_mime}
            offset += len(audio)
    head = dict(header, version=VERSION, frames=len(index), index=index, audio=audio_info)
    hb = json.dumps(head, separators=(",", ":")).encode("utf-8")
    pad = (-(12 + len(hb))) % 8
    with open(path, "wb") as out, open(tmp, "rb") as f:
        out.write(MAGIC + struct.pack("<II", VERSION, len(hb)) + hb + b"\0" * pad)
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            out.write(b)
    os.remove(tmp)
    return head


class Reader:
    """Random access to a .splv file: header, frames (decoded sequentially from the nearest keyframe),
    the audio track."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            magic, ver, hl = f.read(4), *struct.unpack("<II", f.read(8))
            if magic != MAGIC:
                raise ValueError("not a .splv splat video")
            if ver > VERSION:
                raise ValueError(f"made by a newer version (format {ver})")
            self.header = json.loads(f.read(hl).decode("utf-8"))
        self.data_start = 12 + hl + ((-(12 + hl)) % 8)
        self.count = int(self.header["count"])
        self.frames = int(self.header["frames"])
        self.fps = float(self.header["fps"])
        self._cache_k, self._cache = -1, None

    def _chunk(self, k):
        off, size, _ = self.header["index"][k]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + off)
            return f.read(size)

    def quantized(self, k: int) -> dict:
        k = max(0, min(int(k), self.frames - 1))
        if k == self._cache_k:
            return self._cache
        if self._cache is not None and k == self._cache_k + 1 and not self.header["index"][k][2]:
            cur = decode_frame(self._chunk(k), self.count, self._cache)
        else:
            start = k
            while start > 0 and not self.header["index"][start][2]:
                start -= 1
            cur = None
            for j in range(start, k + 1):
                cur = decode_frame(self._chunk(j), self.count, None if self.header["index"][j][2] else cur)
        self._cache_k, self._cache = k, cur
        return cur

    def frame(self, k: int) -> dict:
        return dequantize(self.quantized(k), self.header["quant"])

    def audio(self) -> bytes | None:
        a = self.header.get("audio")
        if not a:
            return None
        with open(self.path, "rb") as f:
            f.seek(self.data_start + a["offset"])
            return f.read(a["size"])
