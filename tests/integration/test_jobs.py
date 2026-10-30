"""End-to-end job pipeline with TEST-ONLY fake workers (tests/fakes/): Real 3D for photos (fake MoGe, fake
SHARP with depth, fake renderer, fake fit) and splat videos for videos (fake per-frame worker). Exercises
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


def test_maximum_detail_for_photos(ctx, rtx4080, photo):
    """Quality means maximum detail; the worker gets the full photo; the normal result is kept for comparison in
    the views sheet. Auto and Fast do not use it."""
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["detail"] is True and req["photo"].endswith("frame_00000.png") and req["moge_path"].endswith("model.pt")
    ex = rep["backend"]["extra"]
    assert ex["detail"]["splats"] == ex["gaussians"] and Path(rep["outputs"]["compare_normal_detail"]).exists()
    std = load_scene(Path(rep["outputs"]["compare_normal_detail"]))
    assert ex["gaussians"] == 2 * len(std)
    for mode in ("auto", "fast"):
        job, rep, _ = run_job(ctx, rtx4080, photo, mode=mode)
        req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
        assert req["detail"] is False and "compare_normal_detail" not in rep["outputs"]


def test_photo_modes_do_different_things(ctx, rtx4080, photo):
    """rc9 (owner: hardly any difference between Fast, Auto and Quality for photos): each mode now runs a
    different plan (backends/locked.PHOTO_PLANS) - Fast skips the slow steps, Quality adds the most."""
    from conftest import install_fake_model

    for mid in ("depth-pro", "sapiens2-seg-0.4b", "sapiens2-pointmap-1b", "sapiens2-normal-1b", "moge-3-vitg"):
        install_fake_model(ctx.models, mid)
    seen = {}
    for mode in ("fast", "auto", "quality"):
        job, rep, _ = run_job(ctx, rtx4080, photo, mode=mode)
        assert rep["status"] == "succeeded", rep.get("error")
        geo = json.loads((job.dir / "worker" / "fake_moge_worker_request.json").read_text())
        req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
        fit = job.dir / "locked" / "fake_refine_request.json"
        seen[mode] = {"giant": "moge-3-vitg" in geo["model_path"], "passes": geo["precision"]["passes"],
                      "depth_pro": bool(geo["precision"]["depth_pro"]), "normals": req["normals"],
                      "people_normals": bool(req["human_normal"]), "detail": req["detail"], "compare": req["plain"],
                      "fit": json.loads(fit.read_text()) if fit.exists() else None,
                      "mode": rep["backend"]["extra"]["mode"], "estimate": rep["estimate"]["seconds"]}
    f, a, q = seen["fast"], seen["auto"], seen["quality"]
    assert not any((f["giant"], f["passes"], f["depth_pro"], f["normals"], f["people_normals"], f["detail"],
                    f["compare"])) and f["fit"] is None
    assert all((a["giant"], a["passes"], a["depth_pro"], a["normals"], a["people_normals"])) and not a["detail"]
    assert a["fit"]["iters"] == 300 and a["fit"]["max_side"] == 2048
    assert q["detail"] and q["giant"] and q["fit"]["iters"] == 600 and q["fit"]["max_side"] == 2560
    assert f["estimate"] < a["estimate"] < q["estimate"] and [seen[m]["mode"] for m in seen] == list(seen)


def test_precision_for_photos(ctx, rtx4080, photo, monkeypatch):
    """rc7: the lens check, several MoGe passes, Depth Pro, people's normals, the fit to the photo and the
    relief sheet reach the workers and the report; MoGe-3 Giant is used for photos when installed."""
    from conftest import install_fake_model

    for mid in ("depth-pro", "sapiens2-seg-0.4b", "sapiens2-pointmap-1b", "sapiens2-normal-1b", "moge-3-vitg"):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="auto")
    assert rep["status"] == "succeeded", rep.get("error")
    geo = json.loads((job.dir / "worker" / "fake_moge_worker_request.json").read_text())
    assert geo["precision"]["passes"] and geo["precision"]["depth_pro"].endswith("depth_pro.pt")
    assert "moge-3-vitg" in geo["model_path"] and geo["version"] == "v3"
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["precision"] and req["human_normal"].endswith("sapiens2_1b_normal.safetensors")
    assert req["refine"] is False                                    # the fit runs in the renderer runtime
    fit = json.loads((job.dir / "locked" / "fake_refine_request.json").read_text())
    assert fit["ply"].endswith("locked.ply") and fit["photo"].endswith("frame_00000.png") and len(fit["size"]) == 2
    assert "moge-3-vitg" in req["moge_path"]
    ex = rep["backend"]["extra"]
    assert ex["precision"]["geometry"]["passes"] == 3 and ex["precision"]["photo"]["fit"]["used"]
    assert Path(rep["outputs"]["relief"]).exists()
    assert {"depth-pro", "sapiens2-normal-1b", "moge-3-vitg"} <= {m["id"] for m in rep["models"]}
    assert "Depth Pro" in rep["coverage"]["note"]
    # an EXIF lens both depth models disagree with is replaced, and said so
    monkeypatch.setenv("FAKE_LENS_OFF", "1")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast")
    assert any("lens" in w and "disagrees" in w for w in rep["warnings"])
    # without the optional models the precision path still runs on MoGe alone
    for mid in ("depth-pro", "sapiens2-normal-1b", "moge-3-vitg"):
        ctx.models.delete(mid)
    monkeypatch.delenv("FAKE_LENS_OFF")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="auto")
    assert rep["status"] == "succeeded"
    geo = json.loads((job.dir / "worker" / "fake_moge_worker_request.json").read_text())
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert geo["precision"]["depth_pro"] is None and "moge-3-vitl" in geo["model_path"]
    assert req["human_normal"] is None and req["precision"]


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
    o = JobOptions.from_dict({"mode": "fast", "vr180": True, "layouts": ["sbs"], "ipd_mm": 60,
                              "real3d_fill": True, "photo_detail": True})
    assert o.mode == "fast" and o.ipd_mm == 60 and not hasattr(o, "vr180") and not hasattr(o, "real3d_fill")


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


def test_video_modes_do_different_things(ctx, rtx4080, videos):
    """rc10: Fast, Auto and Quality differ for videos as for photos (backends/video_splat.PLANS): every mode
    cleans depth edges and places every splat; each step up adds splats, depth resolution and refinement, and
    Quality the fine relief from normals."""
    seen = {}
    for mode in ("fast", "auto", "quality"):
        job, rep, _ = run_job(ctx, rtx4080, videos["static"], mode=mode, video_fps=5, video_seconds=1.0)
        assert rep["status"] == "succeeded", rep.get("error")
        req = json.loads((job.dir / "splat_video" / "fake_requests.json").read_text())[0]
        seen[mode] = req
        assert req["edges"] and req["place_all"] and req["people"]
        assert rep["backend"]["extra"]["mode"] == mode
    f, a, q = seen["fast"], seen["auto"], seen["quality"]
    assert f["moge_max_side"] < a["moge_max_side"] < q["moge_max_side"]
    assert f["refine_steps"] < a["refine_steps"] < q["refine_steps"]
    assert f["grid"][0] > a["grid"][0] > q["grid"][0]                        # more splats per frame
    assert not f["normals"] and not a["normals"] and q["normals"]


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
