"""End-to-end job pipeline with TEST-ONLY fake workers (tests/fakes/): Real 3D for photos (fake MoGe, fake
SHARP with depth, fake renderer, fake painter) and splat videos for videos (fake per-frame worker). Exercises
the real worker protocol, selection, export, validation, reports and error paths."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from twod2vr.jobs import Job, JobOptions, JobRunner
from twod2vr.scene import load_scene

REQUIRED_REPORT_KEYS = {"input", "backend", "models", "hardware", "vram_peak_mib", "runtime_s", "outputs",
                        "coverage", "warnings", "selection", "timings_s", "status", "app"}
QWEN_IDS = ("qwen-image-edit-2511-q5", "qwen-image-edit-2511-base", "qwen-edit-2511-lightning")


def run_job(ctx, hw, path, **kw):
    job = Job(Path(path), JobOptions(**kw))
    events = []
    rep = JobRunner(ctx, hw).run(job, events.append)
    assert (job.dir / "run_report.json").exists()
    assert json.loads((job.dir / "run_report.json").read_text())["status"] == rep["status"]
    return job, rep, events


# ------------------------------------------------------------------ photos: Real 3D

def test_photo_end_to_end(ctx, rtx4080, tmp_path):
    """Real 3D = SHARP aligned to the photo's measured depth (MoGe-3) at its real lens."""
    from PIL import Image

    from conftest import install_fake_model, synthetic_image

    install_fake_model(ctx.models, "yunet-face")
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26
    p = tmp_path / "phone.jpg"
    Image.fromarray(synthetic_image(seed=4)).save(p, exif=exif)
    job, rep, events = run_job(ctx, rtx4080, p, mode="auto")
    assert rep["status"] == "succeeded", rep.get("error")
    assert REQUIRED_REPORT_KEYS <= set(rep)
    assert rep["backend"]["id"] == "photo_locked" and rep["app"]["name"] == "2D2VR"
    assert rep["backend"]["upstream"]["sharp"]["commit"]
    geo = json.loads((job.dir / "worker" / "fake_moge_worker_request.json").read_text())
    assert geo["version"] == "v3" and 55 < geo["images"][0]["fov_x"] < 75          # MoGe-3 at the EXIF lens
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["geometry"].endswith("photo.npz") and req["face_model"].endswith(".onnx") and req["plain"]
    ex = rep["backend"]["extra"]
    assert ex["assembly"] == "locked" and ex["sharp_own_scale"] == 1.2 and ex["face_scale"] == 1.0
    out = rep["outputs"]
    sc = load_scene(Path(out["scene_ply"]))
    assert len(sc) > 1000 and sc.validate() == []
    assert Path(out["splat"]).stat().st_size == 32 * len(sc)
    assert Path(out["views_sheet"]).exists() and Path(out["compare_sharp_alone"]).exists()
    assert {"sharp", "moge-3-vitl", "yunet-face"} <= {m["id"] for m in rep["models"]}
    assert rep["coverage"]["by_splat"]["inferred"] == 1.0
    assert rep["validation"]["source_view"] in ("ok", "degraded")
    assert "vr180" not in out
    readme = (job.dir / "export" / "README.txt").read_text(encoding="utf-8")
    assert "scene.ply" in readme and "VR180" not in readme and "A / X smaller" in readme
    assert events[-1]["event"] == "finished"
    prog = [e["value"] for e in events if e["event"] == "progress"]
    assert prog == sorted(prog) and prog[-1] == 1.0


def test_real3d_fills_what_the_photo_hides_from_moved_views(ctx, rtx4080, photo):
    """Renders from moved cameras → holes (what the photo never saw) → Qwen paints only the holes → SHARP with
    the depth tied to the scene → only the holes' splats are added (GENERATIVE)."""
    from conftest import install_fake_model
    from twod2vr.scene import read_gaussian_ply

    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    calls = json.loads((job.dir / "locked" / "fill" / "painted" / "stage_calls.json").read_text())
    assert [c["stage"] for c in calls] == ["encode", "generate"]
    views = calls[1]["views"]
    assert calls[1]["angles_strength"] == 0 and all(v["prompt"].startswith("The pure black areas") for v in views)
    assert all(v["image"].endswith("_holes.png") for v in views) and 1 <= len(views) <= 4
    fill = json.loads((job.dir / "locked" / "fill" / "fake_fill_request.json").read_text())
    assert len(fill["views"]) == len(views) and fill["moge_version"] == "v3"
    ex = rep["backend"]["extra"]
    assert ex["added_splats"] > 0 and ex["fill"][-1]["view"] == "photo"
    out = rep["outputs"]
    assert Path(out["fill_steps"]).exists() and Path(out["added_layer"]).exists()
    added = read_gaussian_ply(Path(out["added_layer"]))
    assert len(added) == ex["added_splats"] and (added.provenance == 2).all()
    assert rep["coverage"]["by_splat"]["generative"] > 0
    assert set(QWEN_IDS) <= {m["id"] for m in rep["models"]}


def test_real3d_fill_can_be_turned_off(ctx, rtx4080, photo):
    from conftest import install_fake_model

    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", real3d_fill=False)
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["extra"]["added_splats"] == 0 and not (job.dir / "locked" / "fill").exists()


def test_real3d_without_the_painter_says_so(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    assert "needs Qwen-Image-Edit" in (job.dir / "job.log").read_text(encoding="utf-8")


def test_real3d_uses_the_human_model_and_the_eye_distance_setting(ctx, rtx4080, photo):
    """With Sapiens2 installed the worker gets its models, the eye distance comes from the option, and without
    faces the scale comes from the human model."""
    from conftest import install_fake_model

    for mid in ("sapiens2-seg-0.4b", "sapiens2-pointmap-1b"):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", ipd_mm=56.0)
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["human_seg"].endswith("sapiens2_0.4b_seg.safetensors")
    assert req["human_pointmap"].endswith("sapiens2_1b_pointmap.safetensors")
    assert req["ipd_target_m"] == pytest.approx(0.056)
    ex = rep["backend"]["extra"]
    assert ex["scale_from"] == "human model" and ex["people"]["human_model"] == "sapiens2_1b"
    assert {"sapiens2-seg-0.4b", "sapiens2-pointmap-1b"} <= {m["id"] for m in rep["models"]}


def test_real3d_without_the_human_model_says_so(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["human_seg"] is None and req["ipd_target_m"] == pytest.approx(0.063)
    assert "needs Sapiens2" in (job.dir / "job.log").read_text(encoding="utf-8")


def test_commercial_profile_blocks_noncommercial_models(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, license_profile="commercial")
    assert rep["status"] == "failed" and rep["error"]["code"] == "no_backend"
    assert "non-commercial" in rep["error"]["message"]


def test_cpu_mode_without_gpu(ctx, no_gpu, photo):
    job, rep, _ = run_job(ctx, no_gpu, photo)
    assert rep["status"] == "failed" and rep["error"]["code"] == "no_backend"
    assert "VRAM" in rep["error"]["message"]


def test_missing_model(ctx, rtx4080, photo):
    ctx.models.delete("moge-3-vitl")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality")
    assert rep["status"] == "failed" and rep["error"]["code"] == "no_backend"
    assert "MoGe-3" in rep["error"]["message"]


def test_corrupted_input(ctx, rtx4080, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    job, rep, _ = run_job(ctx, rtx4080, bad)
    assert rep["status"] == "failed" and rep["error"]["code"] == "bad_input"


def test_unsupported_input(ctx, rtx4080, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("x")
    job, rep, _ = run_job(ctx, rtx4080, p)
    assert rep["status"] == "failed"


def test_worker_oom_reported(ctx, rtx4080, photo, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_FAIL", "oom")
    job, rep, _ = run_job(ctx, rtx4080, photo)
    assert rep["status"] == "failed" and rep["error"]["code"] == "oom"


def test_job_state_is_final_only_after_report_is_written(ctx, rtx4080, photo, monkeypatch):
    """Regression (Windows CI): the GUI saw state 'succeeded' before run_report.json existed."""
    import twod2vr.jobs as jobs_mod

    job = Job(photo, JobOptions(mode="fast"))
    seen = []
    real_write = Path.write_text

    def spy(self, *a, **k):
        if self.name == "run_report.json":
            seen.append(job.state)  # state at the moment the report is written
        return real_write(self, *a, **k)

    monkeypatch.setattr(jobs_mod.Path, "write_text", spy)
    rep = JobRunner(ctx, rtx4080).run(job)
    assert rep["status"] == "succeeded"
    assert seen == ["running"] and job.state == "succeeded" and job.report is rep


def test_old_options_are_ignored():
    o = JobOptions.from_dict({"mode": "fast", "vr180": True, "layouts": ["sbs"], "ipd_mm": 60})
    assert o.mode == "fast" and o.ipd_mm == 60 and not hasattr(o, "vr180")


# ------------------------------------------------------------------ videos: splat videos

def _video_with_sound(tmp_path, seconds=3.0, fps=15):
    """A short video with a moving red square and a 440 Hz tone."""
    import subprocess

    from conftest import textured_world, write_video
    from twod2vr.media import find_ffmpeg

    world = textured_world()
    frames = []
    for i in range(int(seconds * fps)):
        f = world[:, 100:260].copy()
        f[40:80, 10 + i * 2: 40 + i * 2] = [255, 0, 0]
        frames.append(f)
    silent = write_video(tmp_path / "silent.mp4", frames, fps)
    out = tmp_path / "clip.mp4"
    subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-f", "lavfi",
                    "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "copy", "-c:a", "aac", "-shortest",
                    str(out)], check=True)
    return out


def test_video_becomes_a_splat_video_with_sound(ctx, rtx4080, tmp_path):
    from twod2vr.splv import frame_scene, open_reader

    clip = _video_with_sound(tmp_path)
    job, rep, events = run_job(ctx, rtx4080, clip, mode="fast", video_fps=10)
    assert rep["status"] == "succeeded", rep.get("error")
    assert REQUIRED_REPORT_KEYS <= set(rep)
    assert rep["backend"]["id"] == "splat_video"
    assert rep["processing"]["video_fps"] == 10 and rep["processing"]["frames"] == 30 and rep["processing"]["audio"]
    out = rep["outputs"]
    assert out["frames"] == 30 and out["fps"] == 10 and out["audio"] is True
    r = open_reader(out["splv"])
    assert r.frames == 30 and r.fps == 10 and r.count == 160 * 240
    assert r.header["audio"]["mime"].startswith("audio/webm") and len(r.audio()) > 1000
    keys = [k for k, (_, _, key) in enumerate(r.header["index"]) if key]
    assert keys == [0, 10, 20]                                  # a keyframe every second (seeking)
    # frames decode in order and at random; the red square moves right
    def red_x(k):
        f = r.frame(k)
        red = (f["colors"][:, 0] > 0.8) & (f["colors"][:, 1] < 0.3)
        return float(f["means"][red, 0].mean())
    assert red_x(29) > red_x(0) + 0.1
    first = load_scene(Path(out["scene_ply"]))
    assert len(first) == r.count and np.allclose(first.means, frame_scene(r, 0).means, atol=1e-5)
    req = json.loads((job.dir / "splat_video" / "fake_requests.json").read_text())[0]
    assert req["grid"] == [1.5, 3.0] and req["keyframe_interval"] == 10 and req["people"] is True
    assert req["checkpoint"].endswith("sharp_2572gikvuh.pt") and req["moge_path"].endswith("model.pt")
    assert not (job.dir / "splat_video" / "frames").exists()     # the frames now live in scene.splv
    assert not (job.dir / "frames").exists() and not list(job.dir.glob("audio.*"))   # cleaned after success
    assert "scene.splv" in (job.dir / "export" / "README.txt").read_text(encoding="utf-8")
    prog = [e["value"] for e in events if e["event"] == "progress"]
    assert prog == sorted(prog) and prog[-1] == 1.0


def test_video_options_choose_the_part_of_the_video(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", video_start_s=1.0, video_seconds=1.0,
                          video_fps=5)
    assert rep["status"] == "succeeded", rep.get("error")
    pr = rep["processing"]
    assert pr["video_start_s"] == 1.0 and pr["frames"] == 5 and pr["video_fps"] == 5
    assert rep["outputs"]["audio"] is False
    assert any("no sound" in w for w in rep["warnings"])


def test_video_rate_is_capped_by_the_videos_own(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["static"], mode="fast", video_fps=30)
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["processing"]["video_fps"] == 15 and rep["processing"]["frames"] == 45


def test_out_of_memory_resumes_with_lighter_settings(ctx, rtx4080, videos, monkeypatch):
    monkeypatch.setenv("FAKE_VIDEO_OOM_AT", "7")
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", video_fps=5)
    assert rep["status"] == "succeeded", rep.get("error")
    reqs = json.loads((job.dir / "splat_video" / "fake_requests.json").read_text())
    assert [r["people"] for r in reqs] == [True, False]
    assert any("ran out of memory" in w for w in rep["warnings"])
    from twod2vr.splv import open_reader

    r = open_reader(rep["outputs"]["splv"])
    assert r.frames == 15 and r.header["index"][7][2]           # the resumed part starts with a keyframe


def test_cancel_kills_the_video_worker(ctx, rtx4080, videos, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_DELAY", "1")
    job = Job(videos["dynamic"], JobOptions(mode="fast"))
    runner = JobRunner(ctx, rtx4080)

    def on_event(ev):
        if ev["event"] == "progress" and "reconstructing" in ev.get("message", ""):
            threading.Timer(0.5, job.cancel).start()

    rep = runner.run(job, on_event)
    assert rep["status"] == "cancelled"
    assert rep["runtime_s"] < 30
    assert not (job.dir / "frames").exists()            # a video's frames do not stay behind


def test_not_enough_disk_for_a_splat_video(ctx, rtx4080, videos, monkeypatch):
    import shutil as sh

    from twod2vr.backends import video_splat

    monkeypatch.setattr(video_splat.shutil, "disk_usage", lambda p: sh._ntuple_diskusage(10**12, 10**12 - 10**6, 10**6))
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast")
    assert rep["status"] == "failed" and rep["error"]["code"] == "low_disk"
    assert "shorter part" in rep["error"]["message"]


def test_several_inputs_in_one_job_are_refused(ctx, rtx4080, photo, tmp_path):
    job = Job([photo, photo], JobOptions())
    rep = JobRunner(ctx, rtx4080).run(job)
    assert rep["status"] == "failed" and "own jobs" in rep["error"]["message"]
