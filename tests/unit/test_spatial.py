import subprocess

import numpy as np
import pytest

from twod2vr180.media import find_ffmpeg, iter_video_frames
from twod2vr180.spatial import SpatialError, inject, read_metadata
from twod2vr180.vr180 import StereoOptions, encode_video


def _encode(tmp_path, opts, n=6):
    ff = find_ffmpeg()
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 128, 3), dtype=np.uint8) for _ in range(n)]
    return encode_video(iter(frames), 128, 64, 10, tmp_path / "v.mp4", opts, ff), frames


@pytest.mark.parametrize("layout", ["sbs", "tb"])
def test_vr180_mp4_carries_spherical_v2(tmp_path, layout):
    meta, frames = _encode(tmp_path, StereoOptions(layout=layout))
    assert "error" not in meta["spherical_metadata"]
    md = read_metadata(tmp_path / "v.mp4")
    assert md["stereo_mode"] == layout and md["projection"] == "equirect180"
    # FFmpeg itself recognises the side data and still decodes every frame
    probe = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(tmp_path / "v.mp4")],
                           capture_output=True, text=True).stderr
    assert "spherical" in probe.lower()
    assert ("side by side" if layout == "sbs" else "top and bottom") in probe.lower()
    decoded = list(iter_video_frames(tmp_path / "v.mp4"))
    assert len(decoded) == len(frames)


def test_flat_stereo_gets_st3d_only(tmp_path):
    _encode(tmp_path, StereoOptions(layout="sbs", projection="flat"))
    md = read_metadata(tmp_path / "v.mp4")
    assert md == {"stereo_mode": "sbs"}


def test_double_injection_refused(tmp_path):
    _encode(tmp_path, StereoOptions())
    with pytest.raises(SpatialError):
        inject(tmp_path / "v.mp4", "sbs")
