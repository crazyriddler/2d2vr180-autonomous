"""Inject Spherical Video V2 metadata (st3d + sv3d) into an MP4.

Spec: https://github.com/google/spatial-media/blob/master/docs/spherical-video-v2-rfc.md
Boxes go inside the video track's visual sample entry (avc1/hvc1/...):

    st3d  full box: stereo_mode u8 (0 mono, 1 top-bottom, 2 left-right)
    sv3d  container
      svhd  full box: metadata_source (null-terminated string)
      proj  container
        prhd  full box: yaw, pitch, roll (16.16 fixed, degrees)
        equi  full box: bounds top, bottom, left, right (0.32 fixed fractions cropped)

VR180 = equirectangular with a quarter of the 360° width cropped on each side.
Inserting bytes into ``moov`` shifts the media data when ``moov`` precedes
``mdat`` (faststart), so every stco/co64 chunk offset is adjusted.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}
VISUAL_ENTRIES = {b"avc1", b"avc3", b"hvc1", b"hev1", b"mp4v", b"av01", b"vp09"}
VISUAL_ENTRY_HEADER = 78  # bytes after the 8-byte box header before child boxes


class SpatialError(Exception):
    pass


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full(kind: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _box(kind, struct.pack(">I", (version << 24) | flags) + payload)


def spherical_boxes(layout: str, projection: str = "equirect180") -> bytes:
    stereo = {"mono": 0, "tb": 1, "sbs": 2}[layout]
    out = _full(b"st3d", struct.pack(">B", stereo))
    if projection in ("equirect180", "equirect360"):
        side = 0x40000000 if projection == "equirect180" else 0
        svhd = _full(b"svhd", b"2D2VR180\x00")
        prhd = _full(b"prhd", struct.pack(">iii", 0, 0, 0))
        equi = _full(b"equi", struct.pack(">IIII", 0, 0, side, side))
        out += _box(b"sv3d", svhd + _box(b"proj", prhd + equi))
    return out


def _iter_boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            raise SpatialError(f"corrupt box {kind!r} at {pos}")
        yield pos, size, hdr, kind
        pos += size


def _find_video_entries(buf: bytes, start: int, end: int, path: list[tuple[int, int]]):
    """Yield (entry_pos, entry_size, ancestors) for visual sample entries."""
    for pos, size, hdr, kind in _iter_boxes(buf, start, end):
        if kind in CONTAINERS:
            yield from _find_video_entries(buf, pos + hdr, pos + size, path + [(pos, hdr)])
        elif kind == b"stsd":
            count = struct.unpack(">I", buf[pos + hdr + 4:pos + hdr + 8])[0]
            p = pos + hdr + 8
            for _ in range(count):
                esize, ekind = struct.unpack(">I4s", buf[p:p + 8])
                if ekind in VISUAL_ENTRIES:
                    yield p, esize, path + [(pos, hdr)]
                p += esize


def _set_size(buf: bytearray, pos: int, hdr: int, delta: int) -> None:
    if hdr == 16:
        size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
        buf[pos + 8:pos + 16] = struct.pack(">Q", size + delta)
    else:
        size = struct.unpack(">I", buf[pos:pos + 4])[0]
        if size + delta > 0xFFFFFFFF:
            raise SpatialError("box too large to grow")
        buf[pos:pos + 4] = struct.pack(">I", size + delta)


def _shift_chunk_offsets(moov: bytearray, start: int, end: int, delta: int) -> None:
    for pos, size, hdr, kind in _iter_boxes(moov, start, end):
        if kind in CONTAINERS:
            _shift_chunk_offsets(moov, pos + hdr, pos + size, delta)
        elif kind in (b"stco", b"co64"):
            n = struct.unpack(">I", moov[pos + hdr + 4:pos + hdr + 8])[0]
            p = pos + hdr + 8
            fmt, width = (">I", 4) if kind == b"stco" else (">Q", 8)
            for _ in range(n):
                v = struct.unpack(fmt, moov[p:p + width])[0] + delta
                if kind == b"stco" and v > 0xFFFFFFFF:
                    raise SpatialError("chunk offset overflow; re-encode without faststart")
                moov[p:p + width] = struct.pack(fmt, v)
                p += width


def inject(path: Path, layout: str, projection: str = "equirect180") -> dict:
    """Rewrite ``path`` in place (atomically) with spherical metadata."""
    path = Path(path)
    data = path.read_bytes()
    top = list(_iter_boxes(data, 0, len(data)))
    moov = next(((p, s, h) for p, s, h, k in top if k == b"moov"), None)
    if moov is None:
        raise SpatialError("no moov box")
    mpos, msize, mhdr = moov
    mdat_after = any(k == b"mdat" and p > mpos for p, s, h, k in top)
    new_moov = bytearray(data[mpos:mpos + msize])
    entries = list(_find_video_entries(bytes(new_moov), mhdr, msize, [(0, mhdr)]))
    if not entries:
        raise SpatialError("no video sample entry found")
    payload = spherical_boxes(layout, projection)
    # insert from the last entry backwards so earlier offsets stay valid
    for epos, esize, ancestors in sorted(entries, key=lambda e: -e[0]):
        body = bytes(new_moov[epos + 8 + VISUAL_ENTRY_HEADER: epos + esize])
        if b"st3d" in body or b"sv3d" in body:
            raise SpatialError("file already carries spherical metadata")
        insert_at = epos + esize
        new_moov[insert_at:insert_at] = payload
        _set_size(new_moov, epos, 8, len(payload))
        for apos, ahdr in ancestors:
            _set_size(new_moov, apos, ahdr, len(payload))
    delta = len(new_moov) - msize
    if mdat_after:
        _shift_chunk_offsets(new_moov, mhdr, len(new_moov), delta)
    out = data[:mpos] + bytes(new_moov) + data[mpos + msize:]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(out)
    os.replace(tmp, path)
    return {"standard": "Spherical Video V2 (st3d/sv3d)", "stereo_mode": layout, "projection": projection,
            "inserted_bytes": delta * len(entries) // max(len(entries), 1)}


def read_metadata(path: Path) -> dict:
    """Parse st3d/sv3d back (for validation)."""
    data = Path(path).read_bytes()
    out: dict = {}
    top = list(_iter_boxes(data, 0, len(data)))
    moov = next(((p, s, h) for p, s, h, k in top if k == b"moov"), None)
    if moov is None:
        return out
    mpos, msize, mhdr = moov
    blob = data[mpos:mpos + msize]
    for epos, esize, _ in _find_video_entries(blob, mhdr, msize, [(0, mhdr)]):
        start = epos + 8 + VISUAL_ENTRY_HEADER
        for pos, size, hdr, kind in _iter_boxes(blob, start, epos + esize):
            if kind == b"st3d":
                out["stereo_mode"] = {0: "mono", 1: "tb", 2: "sbs"}.get(blob[pos + hdr + 4])
            elif kind == b"sv3d":
                i = blob.find(b"equi", pos, pos + size)
                if i > 0:
                    t, b, l, r = struct.unpack(">IIII", blob[i + 8:i + 24])
                    out["equi_bounds"] = {"top": t, "bottom": b, "left": l, "right": r}
                    out["projection"] = "equirect180" if l == r == 0x40000000 else "equirect"
    return out
