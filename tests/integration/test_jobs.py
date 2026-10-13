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
    job = Job(path if isinstance(path, list) else Path(path), JobOptions(**kw))
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
    job = Job(videos["dynamic"], JobOptions(eye_resolution=128, video_eye_resolution=64, mode="fast"))
    runner = JobRunner(ctx, rtx4080)

    def on_event(ev):
        if ev["event"] == "progress" and "reconstructing" in ev.get("message", ""):
            threading.Timer(0.5, job.cancel).start()

    rep = runner.run(job, on_event)
    assert rep["status"] == "cancelled"
    assert rep["runtime_s"] < 30


def test_static_video_becomes_a_4d_scene_with_nothing_moving(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["static"], mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["input"]["analysis"]["kind"] == "static_scene"
    assert rep["backend"]["id"] == "video4d"
    assert len(list((job.dir / "frames").glob("*.png"))) == 24        # fast: 8 per second of the 3 s video
    assert rep["outputs"]["frames"] == 24 and rep["outputs"]["moving_gaussians"] == 0


def test_dynamic_video_becomes_a_4d_scene_and_a_4d_vr180_video(ctx, rtx4080, videos):
    from twod2vr180.scene4d import Scene4D, read_splatv

    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    out = rep["outputs"]
    assert out["frames"] == 24 and out["fps"] == 8.0 and out["moving_gaussians"] > 0
    s4 = Scene4D.load(Path(out["scene4d"]))
    assert s4.frames == 24 and len(s4.dynamic) == out["moving_gaussians"]
    # the red box moves to the right over the clip
    x0 = s4.dynamic.at(s4.time_of(0)).means[:, 0].mean()
    x1 = s4.dynamic.at(s4.time_of(23)).means[:, 0].mean()
    assert x1 > x0 + 0.3
    v = read_splatv(Path(out["splatv"]))
    assert v.frames == 24 and len(v.background) == len(s4.background) and len(v.dynamic) == len(s4.dynamic)
    vids = out["vr180"]["videos"]
    assert len(vids) == 1 and "_4d_180_LR" in Path(vids[0]["path"]).name
    assert vids[0]["frames"] == 24 and vids[0]["fps"] == 8.0 and "4D scene" in vids[0]["method"]
    assert "fixed camera position" in vids[0]["method"]
    assert rep["validation"]["source_view"] in ("ok", "degraded")
    readme = (job.dir / "export" / "README.txt").read_text()
    assert "scene.splatv" in readme and "4D SCENE" in readme
    assert (job.dir / "export" / "_2d2vr180" / "moving_preview.png").exists()


def test_smooth_4d_motion_plays_continuously_and_renders_vr180_at_24_fps(ctx, rtx4080, videos, monkeypatch):
    from twod2vr180.scene4d import Scene4D

    monkeypatch.setenv("FAKE_4D_MOTION", "1")
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "worker" / "fake_video4d_worker_request.json").read_text())
    assert req["motion"] is True and req["motion_steps"] == 2000
    assert rep["backend"]["extra"]["moving_model"] == "motion"
    s4 = Scene4D.load(Path(rep["outputs"]["scene4d"]))
    assert s4.meta["playback"] == "continuous" and s4.dynamic.motion is not None
    a, b = s4.dynamic.at(0.25).means[:, 0].mean(), s4.dynamic.at(0.75).means[:, 0].mean()
    assert b > a + 0.1                                               # the box travels between frames
    vid = rep["outputs"]["vr180"]["videos"][0]
    assert vid["fps"] == 24.0 and vid["frames"] == 72 and "smoothly" in vid["method"]


def test_video_options_choose_the_part_of_the_video(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", layouts=["sbs"], renderer="cpu",
                          vr180=False, video_seconds=1.0, video_fps=5.0, video_start_s=1.0)
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["outputs"]["frames"] == 5
    assert rep["processing"]["video_fps"] == 5.0 and rep["processing"]["video_start_s"] == 1.0
    assert any("Only 1.0 s of the 3.0 s video" in w for w in rep["warnings"])


def test_video_without_the_4d_engine_falls_back_to_the_sharpest_frame(ctx, rtx4080, videos):
    rtx4080.gpus[0].total_mib = 8_000          # too small for the 4D engine
    job, rep, _ = run_job(ctx, rtx4080, videos["pan"], mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["selection"]["effective_kind"] == "video:frame"
    assert rep["backend"]["id"] == "moge_rgbd"
    assert len(list((job.dir / "frames").glob("*.png"))) == 1
    assert any("falling back" in w and "4D video" in w for w in rep["warnings"])
    assert any(r["backend"] == "video4d" for r in rep["selection"]["rejected"])


def test_moving_camera_video_follows_the_smoothed_camera_path(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["pan"], mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "video4d" and rep["backend"]["extra"]["camera_motion"] == "moving"
    assert "camera path" in rep["outputs"]["vr180"]["videos"][0]["method"]
    assert rep["outputs"]["vr180"]["videos"][0]["spherical_metadata"]["projection"] == "equirect180"


def test_job_state_is_final_only_after_report_is_written(ctx, rtx4080, photo, monkeypatch):
    """Regression (Windows CI): the GUI saw state 'succeeded' before run_report.json existed."""
    import twod2vr180.jobs as jobs_mod

    job = Job(photo, JobOptions(mode="fast", eye_resolution=128, still_video_seconds=0.5, layouts=["sbs"]))
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


def test_several_photos_become_one_multiview_scene(ctx, rtx4080, tmp_path, monkeypatch):
    from PIL import Image

    from conftest import REPO, synthetic_image
    from twod2vr180.backends.multiview import MultiViewBackend

    monkeypatch.setattr(MultiViewBackend, "worker_script", str(REPO / "tests" / "fakes" / "fake_multiview_worker.py"))
    paths = []
    for i in range(4):
        p = tmp_path / f"shot_{i}.jpg"
        Image.fromarray(synthetic_image(seed=i)).save(p)
        paths.append(p)
    job, rep, _ = run_job(ctx, rtx4080, paths, mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["input"]["type"] == "images" and len(rep["input"]["paths"]) == 4
    assert rep["backend"]["id"] == "multiview"
    assert len(list((job.dir / "frames").glob("*.png"))) == 4
    req = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert [Path(v["path"]).name for v in req["images"]][0] == "frame_00000.png"
    assert rep["outputs"]["vr180"]["stills"]


def test_folder_of_photos_is_expanded(tmp_path):
    from twod2vr180.jobs import expand_inputs

    for n in ("b.jpg", "a.png", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in expand_inputs(tmp_path)] == ["a.png", "b.jpg"]


def test_generative_photo_to_3d(ctx, rtx4080, photo, monkeypatch):
    """Explore the surroundings (VR180) with Wan 2.2: every second frame is trained with the photo."""
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    install_fake_model(ctx.models, "wan2.2-fun-5b-camera")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="explore", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "generative_scene"
    assert rep["coverage"]["by_splat"]["generative"] > 0.2
    assert {m["id"] for m in rep["models"]} >= {"wan2.2-fun-5b-camera", "vggt-1b"}
    req = json.loads((job.dir / "worker" / "fake_wan_worker_request.json").read_text())
    assert req["trajectory"] == "explore" and req["frames"] == 49 and req["every"] == 2
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["images"][0]["generated"] is False and mv["images"][0]["weight"] > 1
    assert all(v["generated"] for v in mv["images"][1:])
    assert any("GENERATIVE" in w for w in rep["warnings"])


def test_generative_request_falls_back_when_models_missing(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="capture", layouts=["sbs"])
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "moge_rgbd"
    assert any("Generative 3D was requested" in w for w in rep["warnings"])


def test_vr180_holes_filled_by_lama_when_installed(ctx, rtx4080, photo, monkeypatch):
    from conftest import REPO, install_fake_model
    import twod2vr180.inpaint as inp_mod
    from PIL import Image

    monkeypatch.setattr(inp_mod, "WORKER", str(REPO / "tests" / "fakes" / "fake_inpaint_worker.py"))
    install_fake_model(ctx.models, "big-lama")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", layouts=["sbs"], projection="flat", renderer="cpu",
                          eye_separation_m=0.3)
    assert rep["status"] == "succeeded", rep.get("error")
    still = rep["outputs"]["vr180"]["stills"][0]
    assert "LaMa" in still["metadata"]["hole_filling"]
    img = np.asarray(Image.open(still["image"]).convert("RGB")).astype(int)
    magenta = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)
    assert magenta.any()
    assert not any("LaMa) not used" in w for w in rep["warnings"])


def test_real3d_uses_wan_capture_when_qwen_is_missing(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    install_fake_model(ctx.models, "wan2.2-fun-5b-camera")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["extra"]["engine"] == "wan" and rep["backend"]["extra"]["trajectory"] == "capture"
    req = json.loads((job.dir / "worker" / "fake_wan_worker_request.json").read_text())
    assert req["trajectory"] == "capture" and req["frames"] == 33 and req["steps"] == 25
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert len(mv["images"]) > 2 and not mv["images"][0]["generated"]       # trained, never fused


def test_wan_out_of_memory_retries_in_a_fresh_process(ctx, rtx4080, photo, monkeypatch):
    from conftest import REPO, install_fake_model
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "wan_script", str(fakes / "fake_wan_worker.py"))
    monkeypatch.setenv("FAKE_WAN_OOM_ATTEMPTS", "2")
    install_fake_model(ctx.models, "wan2.2-fun-5b-camera")
    job, rep, log = run_job(ctx, rtx4080, photo, mode="fast", generative="capture", gen_engine="wan",
                            layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "worker" / "fake_wan_worker_request.json").read_text())
    assert req["attempt"] == 2


def _fake_generative(monkeypatch):
    from conftest import REPO
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "wan_script", str(fakes / "fake_wan_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "qwen_script", str(fakes / "fake_qwen_worker.py"))


QWEN_IDS = ("qwen-image-edit-2511-q5", "qwen-image-edit-2511-base", "qwen-edit-2511-angles-lora",
            "qwen-edit-2511-lightning")


@pytest.mark.parametrize("mode,n_views", [("fast", 7), ("quality", 13)])
def test_qwen_360_capture(ctx, rtx4080, photo, monkeypatch, mode, n_views):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS + ("wan2.2-fun-5b-camera",):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode=mode, generative="capture", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["extra"]["engine"] == "qwen"     # auto prefers Qwen when installed
    gen = job.dir / "generated_views"
    enc = json.loads((gen / "stage_encode.json").read_text())
    angles = [(v["azimuth"], v["elevation"]) for v in enc["views"]]
    assert angles[:7] == [(45, 0), (90, 0), (135, 0), (180, 0), (270, 0), (0, 60), (0, -30)]
    assert len(angles) == n_views
    assert enc["gguf"].endswith("qwen-image-edit-2511-Q5_K_M.gguf") and enc["base_dir"].endswith(
        "qwen-image-edit-2511-base")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert len(mv["images"]) == n_views + 1
    assert {m["id"] for m in rep["models"]} >= set(QWEN_IDS)


def test_wan_360_capture_when_chosen(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS + ("wan2.2-fun-5b-camera",):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality", generative="capture", gen_engine="wan",
                          layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "worker" / "fake_wan_worker_request.json").read_text())
    assert req["trajectory"] == "capture_full" and req["frames"] == 49
    assert rep["backend"]["extra"]["engine"] == "wan"


def test_qwen_cannot_explore_falls_back_to_wan(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS + ("wan2.2-fun-5b-camera",):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="explore", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["extra"]["engine"] == "wan"


@pytest.mark.parametrize("mode", ["fast", "auto", "quality"])
def test_real3d_without_sharp_trains_the_three_views(ctx, rtx4080, photo, monkeypatch, mode):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode=mode, generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert "Apple SHARP is not installed" in (job.dir / "job.log").read_text(encoding="utf-8")
    gen = job.dir / "generated_views"
    calls = json.loads((gen / "stage_calls.json").read_text())
    assert [c["stage"] for c in calls] == ["encode", "generate"]          # one round: fast
    views = calls[1]["views"]
    # exactly 3 views in every mode, one attempt each (owner, rc55)
    assert [(v["azimuth"], v["elevation"]) for v in views] == [(315, 0), (45, 0), (0, 30)]
    assert all(not v.get("refs") for v in views)                            # each drawn from the photo
    assert all(c["candidates"] == 1 and c["angles_strength"] > 0 for c in calls)
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert len(mv["images"]) == 4 and not mv["images"][0]["generated"]
    assert all(not v.get("candidates") for v in mv["images"][1:])
    assert rep["backend"]["extra"]["assembly"] == "train"
    photo_side = {"fast": 1280, "auto": 1600, "quality": 2048}[mode]
    assert mv["images"][0]["max_side"] == photo_side and all(v["max_side"] == 1328 for v in mv["images"][1:])
    assert mv["sh_degree"] == 1 and mv["steps"] == {"fast": 5000, "auto": 9000, "quality": 15000}[mode]


def _fake_sharp(ctx, monkeypatch):
    from conftest import REPO, install_fake_model
    from twod2vr180.backends.generative import GenerativeSceneBackend

    monkeypatch.setattr(GenerativeSceneBackend, "sharp_script", str(REPO / "tests" / "fakes" / "fake_sharp_worker.py"))
    install_fake_model(ctx.models, "sharp")


def test_real3d_gives_the_photo_and_three_views_volume_with_sharp_and_joins_them(ctx, rtx4080, tmp_path,
                                                                                  monkeypatch):
    """The owner's design (rc55): 3 views, one attempt each; SHARP on the photo and the 3 views; one splat."""
    from PIL import Image

    from conftest import install_fake_model, synthetic_image
    from twod2vr180.scene import read_gaussian_ply

    _fake_generative(monkeypatch)
    _fake_sharp(ctx, monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26                     # a phone's 26 mm lens
    p = tmp_path / "phone.jpg"
    Image.fromarray(synthetic_image(seed=3)).save(p, exif=exif)
    job, rep, _ = run_job(ctx, rtx4080, p, mode="auto", generative="tri", real3d_method="views", layouts=["sbs"],
                          renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    log = (job.dir / "job.log").read_text(encoding="utf-8")
    calls = json.loads((job.dir / "generated_views" / "stage_calls.json").read_text())
    assert len(calls[1]["views"]) == 3 and calls[1]["candidates"] == 1
    # SHARP ran once over the photo and the 3 views, each carrying the photo's lens for SHARP to read
    sreq = json.loads((job.dir / "sharp_views" / "fake_sharp_request.json").read_text())
    ins = sorted((Path(sreq["input_dir"])).iterdir())
    assert [f.name for f in ins] == ["00.png", "01.png", "02.png", "03.png"]
    for f in ins:
        with Image.open(f) as im:
            assert 24 < float(im.getexif().get_ifd(0x8769)[0x920A]) < 28
    # the camera engine only placed the cameras: no training
    assert (job.dir / "sharp_views" / "fake_poses_request.json").exists()
    assert not (job.dir / "worker" / "fake_multiview_worker_request.json").exists() or \
        "poses_npz" in json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    ex = rep["backend"]["extra"]
    assert ex["assembly"] == "sharp" and ex["generated_views"] == 3 and ex["poses_from"] == "camera engine"
    assert [r["view"] for r in ex["fusion"]] == ["photo", "45° left", "45° right", "high angle"]
    assert len(ex["view_cameras"]) == ex["views_used"]
    assert {"sharp", "vggt-1b"} <= {m["id"] for m in rep["models"]}
    sc = read_gaussian_ply(job.dir / "export" / "scene.ply")
    assert len(sc) >= 80 * 60 and len(sc.cameras) == 1
    assert "SHARP lens: 2" in log and "photo EXIF" in log


def test_real3d_with_sharp_when_the_camera_engine_fails_uses_the_requested_orbit(ctx, rtx4080, photo, monkeypatch):
    from conftest import FAKE_WORKER, install_fake_model
    from twod2vr180.backends.multiview import MultiViewBackend

    _fake_generative(monkeypatch)
    _fake_sharp(ctx, monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(FAKE_WORKER))     # answers with a bad request
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="tri", real3d_method="views",
                          layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["extra"]["poses_from"] == "requested orbit"
    assert "SHARP lens: 30 mm" in (job.dir / "job.log").read_text(encoding="utf-8")


def test_multiview_backend_on_one_photo_generates_views_first(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", backend="multiview", generative="capture",
                          layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["selection"]["backend"] == "generative_scene"
    assert rep["backend"]["extra"]["assembly"] == "train"


def test_multiview_backend_on_one_photo_without_generation_explains(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", backend="multiview", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "failed" and "generative mode" in rep["error"]["message"]


def test_the_photos_real_lens_reaches_the_reconstruction(ctx, rtx4080, tmp_path, monkeypatch):
    """The job works on PNG copies of the photo (no EXIF): the field of view must be read from the
    original file and travel with the photo's view (rc47 found it was silently lost)."""
    from PIL import Image

    from conftest import install_fake_model, synthetic_image

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26
    p = tmp_path / "phone.jpg"
    Image.fromarray(synthetic_image(seed=3)).save(p, exif=exif)
    job, rep, _ = run_job(ctx, rtx4080, p, mode="fast", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert 55 < mv["images"][0]["fov_x"] < 75 and "fov_x" not in mv["images"][1]


def test_real3d_takes_the_photos_geometry_from_moge3_when_installed(ctx, rtx4080, photo, monkeypatch):
    from conftest import FAKE_WORKER, install_fake_model
    from twod2vr180.backends.generative import MOGE3, GenerativeSceneBackend

    _fake_generative(monkeypatch)
    monkeypatch.setattr(GenerativeSceneBackend, "geometry_script", str(FAKE_WORKER))
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="tri", layouts=["sbs"], renderer="cpu")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert rep["status"] == "succeeded" and "photo_geometry" not in mv        # not installed: MoGe-2 path
    install_fake_model(ctx.models, MOGE3)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["photo_geometry"].endswith("photo.npz") and Path(mv["photo_geometry"]).exists()
    assert MOGE3 in {m["id"] for m in rep["models"]}


def _fake_locked(ctx, monkeypatch):
    from conftest import FAKE_WORKER, REPO, install_fake_model
    from twod2vr180.backends.locked import LockedPhotoBackend

    monkeypatch.setattr(LockedPhotoBackend, "geometry_script", str(FAKE_WORKER))
    monkeypatch.setattr(LockedPhotoBackend, "worker_script", str(REPO / "tests" / "fakes" / "fake_sharp_depth_worker.py"))
    install_fake_model(ctx.models, "sharp")


def test_real3d_locks_the_photo_by_default(ctx, rtx4080, tmp_path, monkeypatch):
    """rc56: Real 3D = SHARP aligned to the photo's measured depth at its real lens; no AI views needed."""
    from PIL import Image

    from conftest import install_fake_model, synthetic_image

    _fake_locked(ctx, monkeypatch)
    install_fake_model(ctx.models, "yunet-face")
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26
    p = tmp_path / "phone.jpg"
    Image.fromarray(synthetic_image(seed=4)).save(p, exif=exif)
    job, rep, _ = run_job(ctx, rtx4080, p, mode="auto", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "photo_locked"
    geo = json.loads((job.dir / "worker" / "fake_moge_worker_request.json").read_text())
    assert geo["version"] == "v2" and 55 < geo["images"][0]["fov_x"] < 75          # MoGe-2 here, at the EXIF lens
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["geometry"].endswith("photo.npz") and req["face_model"].endswith(".onnx") and req["plain"]
    ex = rep["backend"]["extra"]
    assert ex["assembly"] == "locked" and ex["sharp_own_scale"] == 1.2 and ex["face_scale"] == 1.0
    out = rep["outputs"]
    assert Path(out["views_sheet"]).exists() and Path(out["compare_sharp_alone"]).exists()
    from PIL import Image as I

    with I.open(out["views_sheet"]) as im:          # 6 columns (photo + 5 views) x 2 rows
        assert im.width > 5 * im.height / 2
    assert {"sharp", "moge-2-vitl-normal", "yunet-face"} <= {m["id"] for m in rep["models"]}
    assert rep["coverage"]["by_splat"]["inferred"] == 1.0


def test_real3d_locked_uses_moge3_when_installed(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_locked(ctx, monkeypatch)
    install_fake_model(ctx.models, "moge-3-vitl")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    geo = json.loads((job.dir / "worker" / "fake_moge_worker_request.json").read_text())
    assert geo["version"] == "v3"
    req = json.loads((job.dir / "locked" / "fake_sharp_depth_request.json").read_text())
    assert req["face_model"] is None                                                 # not installed: no check


def test_real3d_without_sharp_falls_back_to_the_ai_views_method(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "generative_scene"
    assert "the photo locked) cannot run" in (job.dir / "job.log").read_text(encoding="utf-8")
