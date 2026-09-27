"""End-to-end job pipeline with a TEST-ONLY fake geometry worker (see
tests/fakes/fake_moge_worker.py). Exercises the real worker protocol,
selection, export, VR180, validation, reports and error paths."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from twod2vr180.jobs import Job, JobOptions, JobRunner
from twod2vr180.scene import load_scene

REQUIRED_REPORT_KEYS = {"input", "backend", "models", "hardware", "vram_peak_mib", "runtime_s", "outputs",
                        "coverage", "warnings", "selection", "timings_s", "status", "app"}


def run_job(ctx, hw, path, **kw):
    kw.setdefault("eye_resolution", 256)
    kw.setdefault("video_eye_resolution", 128)
    kw.setdefault("still_video_seconds", 0.5)
    job = Job(Path(path), JobOptions(**kw))
    events = []
    rep = JobRunner(ctx, hw).run(job, events.append)
    assert (job.dir / "run_report.json").exists()
    assert json.loads((job.dir / "run_report.json").read_text())["status"] == rep["status"]
    return job, rep, events


def test_photo_end_to_end(ctx, rtx4080, photo):
    job, rep, events = run_job(ctx, rtx4080, photo, mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    assert REQUIRED_REPORT_KEYS <= set(rep)
    assert rep["backend"]["id"] == "moge_rgbd"
    assert rep["backend"]["upstream"]["moge"]["commit"]
    assert rep["models"][0]["id"] == "moge-2-vits-normal"
    assert rep["coverage"]["by_splat"]["inferred"] == 1.0
    assert rep["validation"]["source_view"] == "ok" and rep["validation"]["psnr_db"] > 25
    out = rep["outputs"]
    sc = load_scene(Path(out["scene_ply"]))
    assert len(sc) > 1000 and sc.validate() == []
    assert Path(out["splat"]).stat().st_size == 32 * len(sc)
    assert Path(out["obj"]).exists()
    stills = out["vr180"]["stills"]
    assert [Path(s["image"]).name for s in stills] == ["photo_180_LR.jpg", "photo_180_TB.jpg"]
    assert stills[0]["metadata"]["width"] == 512 and stills[1]["metadata"]["height"] == 512
    vids = out["vr180"]["videos"]
    assert len(vids) == 2 and all(Path(v["path"]).stat().st_size > 0 for v in vids)
    assert any("not a full hemispherical" in w for w in rep["warnings"])
    assert events[-1]["event"] == "finished"
    prog = [e["value"] for e in events if e["event"] == "progress"]
    assert prog == sorted(prog) and prog[-1] == 1.0


def test_commercial_profile_uses_commercial_safe_backend(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, license_profile="commercial")
    assert rep["status"] == "succeeded" and rep["backend"]["id"] == "moge_rgbd"
    for mid in ("moge-2-vitl-normal", "moge-2-vits-normal"):
        ctx.models.delete(mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, license_profile="commercial")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "depth_anything_v2" and rep["backend"]["commercial_use"] is True
    assert rep["outputs"]["metric_scale"] is False
    assert any("relative" in w for w in rep["warnings"])
    rejected = {r["backend"] for r in rep["selection"]["rejected"]}
    assert {"sharp", "moge_rgbd"} <= rejected


def test_commercial_profile_blocks_noncommercial_models(ctx, rtx4080, photo):
    for mid in ("depth-anything-v2-small", "moge-2-vitl-normal", "moge-2-vits-normal"):
        ctx.models.delete(mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, license_profile="commercial")
    assert rep["status"] == "failed" and rep["error"]["code"] == "no_backend"
    assert "non-commercial" in rep["error"]["message"]


def test_gpu_renderer_failure_falls_back_to_cpu(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", renderer="gpu", layouts=["sbs"])
    assert rep["status"] == "succeeded", rep.get("error")
    assert any("GPU renderer failed" in w for w in rep["warnings"])
    assert rep["outputs"]["vr180"]["stills"][0]["metadata"]["renderer"] == "cpu-reference"


def test_cpu_mode_runs_without_gpu(ctx, no_gpu, photo):
    job, rep, _ = run_job(ctx, no_gpu, photo, mode="fast", allow_cpu=True, renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")


def test_no_gpu_fails_cleanly(ctx, no_gpu, photo):
    job, rep, _ = run_job(ctx, no_gpu, photo)
    assert rep["status"] == "failed" and rep["error"]["code"] == "no_backend"
    assert "VRAM" in rep["error"]["message"]


def test_missing_model(ctx, rtx4080, photo):
    ctx.models.delete("moge-2-vitl-normal")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality")
    assert rep["status"] == "succeeded" and rep["backend"]["id"] == "depth_anything_v2"
    ctx.models.delete("depth-anything-v2-small")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality")
    assert rep["status"] == "failed"
    assert "moge-2-vitl-normal" in rep["error"]["message"]


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


def test_cancel_kills_worker(ctx, rtx4080, tmp_path, videos, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_DELAY", "2")
    job = Job(videos["dynamic"], JobOptions(eye_resolution=128, video_eye_resolution=64, dynamic_fps=5))
    runner = JobRunner(ctx, rtx4080)

    def on_event(ev):
        if ev["event"] == "progress" and "reconstructing" in ev.get("message", ""):
            threading.Timer(0.5, job.cancel).start()

    rep = runner.run(job, on_event)
    assert rep["status"] == "cancelled"
    assert rep["runtime_s"] < 30


def test_static_video_is_treated_as_photo(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["static"], mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["input"]["analysis"]["kind"] == "static_scene"
    assert rep["selection"]["effective_kind"] == "video:static_scene"
    assert len(list((job.dir / "frames").glob("*.png"))) == 1


def test_dynamic_video_per_frame(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", dynamic_fps=5, layouts=["sbs"])
    assert rep["status"] == "succeeded", rep.get("error")
    vids = rep["outputs"]["vr180"]["videos"]
    dyn = [v for v in vids if "dynamic" in Path(v["path"]).name]
    assert dyn and dyn[0]["frames"] >= 10 and dyn[0]["stereo_mode"] == "left_right"
    assert "per-frame" in dyn[0]["method"]


def test_moving_camera_without_multiview_backend_falls_back(ctx, rtx4080, videos):
    # recon3d runtime is "installed" (fake) but its VGGT/MoGe models are runtime-fetched;
    # force unavailability by pretending the GPU is too small for it.
    rtx4080.gpus[0].total_mib = 10_000
    job, rep, _ = run_job(ctx, rtx4080, videos["pan"], mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["selection"]["effective_kind"] == "video:static_scene"
    assert any("falling back" in w for w in rep["warnings"])
    assert any(r["backend"] == "recon3d_video" for r in rep["selection"]["rejected"])


def test_moving_camera_multiview_path_video(ctx, rtx4080, videos, monkeypatch):
    from conftest import REPO
    from twod2vr180.backends.video import Recon3DBackend

    monkeypatch.setattr(Recon3DBackend, "worker_script", str(REPO / "tests" / "fakes" / "fake_recon3d_worker.py"))
    job, rep, _ = run_job(ctx, rtx4080, videos["pan"], mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "recon3d_video"
    assert rep["coverage"]["by_splat"]["observed"] == 1.0
    vids = rep["outputs"]["vr180"]["videos"]
    path = [v for v in vids if "_path_" in Path(v["path"]).name]
    assert path and path[0]["frames"] >= len(rep["input"]["analysis"]["keyframes"])
    assert "interpolated" in path[0]["method"]
    assert path[0]["spherical_metadata"]["projection"] == "equirect180"
