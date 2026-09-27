import numpy as np
import pytest

from twod2vr180.media import (MediaError, analyze_frames, analyze_video, classify_path, load_image,
                              parse_ffmpeg_probe, phase_correlation, probe_video)


def test_classify(tmp_path):
    assert classify_path(tmp_path / "a.JPG") == "photo"
    assert classify_path(tmp_path / "a.webp") == "photo"
    assert classify_path(tmp_path / "a.mov") == "video"
    with pytest.raises(MediaError):
        classify_path(tmp_path / "a.txt")


def test_corrupted_image(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"\xff\xd8\xff garbage")
    with pytest.raises(MediaError):
        load_image(p)


def test_phase_correlation_recovers_shift():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (96, 128)).astype(np.uint8)
    b = np.roll(a, (3, -5), axis=(0, 1))
    dy, dx, _ = phase_correlation(a, b)
    assert (dy, dx) == (3, -5)


def test_parse_probe_rotation():
    txt = ("Duration: 00:00:12.50, start: 0.000000, bitrate: 9000 kb/s\n"
           "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), 1920x1080, "
           "8000 kb/s, 29.97 fps, 29.97 tbr, 90k tbn (default)\n"
           "    Side data:\n      displaymatrix: rotation of -90.00 degrees\n")
    info = parse_ffmpeg_probe(txt)
    assert (info.width, info.height) == (1080, 1920)
    assert info.fps == pytest.approx(29.97) and info.duration_s == pytest.approx(12.5)


def test_static_scene_uses_one_frame(videos):
    an = analyze_video(videos["static"])
    assert an.kind == "static_scene"
    assert len(an.keyframes) == 1 and an.redundant_fraction > 0.9


def test_static_camera_dynamic(videos):
    an = analyze_video(videos["dynamic"])
    assert an.camera_motion == "static"
    assert an.kind == "static_camera_dynamic"


def test_moving_camera(videos):
    an = analyze_video(videos["pan"])
    assert an.kind == "moving_camera"
    assert 3 <= len(an.keyframes) < an.sampled_frames
    assert an.keyframes == sorted(an.keyframes)


def test_scene_cut_detected():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (60, 80)).astype(np.uint8)
    b = rng.integers(0, 255, (60, 80)).astype(np.uint8)
    an = analyze_frames([a] * 10 + [b] * 10, sample_fps=10, src_fps=10)
    assert an.cuts == [10]


def test_corrupted_video(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    with pytest.raises(MediaError):
        probe_video(p)
