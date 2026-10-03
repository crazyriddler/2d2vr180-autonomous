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
    # the multi-view runtime is "installed" (fake): force unavailability with a GPU that is too small.
    rtx4080.gpus[0].total_mib = 8_000
    job, rep, _ = run_job(ctx, rtx4080, videos["pan"], mode="fast")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["selection"]["effective_kind"] == "video:static_scene"
    assert any("falling back" in w for w in rep["warnings"])
    assert any(r["backend"] == "multiview" for r in rep["selection"]["rejected"])


def test_moving_camera_multiview_path_video(ctx, rtx4080, videos, monkeypatch):
    from conftest import REPO
    from twod2vr180.backends.multiview import MultiViewBackend

    monkeypatch.setattr(MultiViewBackend, "worker_script", str(REPO / "tests" / "fakes" / "fake_multiview_worker.py"))
    job, rep, _ = run_job(ctx, rtx4080, videos["pan"], mode="fast", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "multiview"
    assert rep["coverage"]["by_splat"]["observed"] == 1.0
    vids = rep["outputs"]["vr180"]["videos"]
    path = [v for v in vids if "_path_" in Path(v["path"]).name]
    assert path and path[0]["frames"] >= len(rep["input"]["analysis"]["keyframes"])
    assert "interpolated" in path[0]["method"]
    assert path[0]["spherical_metadata"]["projection"] == "equirect180"


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
    tt = rep["backend"]["extra"]["turntable"]
    assert tt and Path(tt).name == "turntable.mp4" and Path(tt).stat().st_size > 0


def test_folder_of_photos_is_expanded(tmp_path):
    from twod2vr180.jobs import expand_inputs

    for n in ("b.jpg", "a.png", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in expand_inputs(tmp_path)] == ["a.png", "b.jpg"]


def test_video_mode_multiview_overrides_static_analysis(ctx, rtx4080, videos, monkeypatch):
    from conftest import REPO
    from twod2vr180.backends.multiview import MultiViewBackend

    monkeypatch.setattr(MultiViewBackend, "worker_script", str(REPO / "tests" / "fakes" / "fake_multiview_worker.py"))
    job, rep, _ = run_job(ctx, rtx4080, videos["static"], mode="fast", layouts=["sbs"], renderer="cpu",
                          video_mode="multiview", max_keyframes=12)
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "multiview"
    assert len(list((job.dir / "frames").glob("*.png"))) >= 6


def test_generative_photo_to_3d(ctx, rtx4080, photo, monkeypatch):
    from conftest import REPO, install_fake_model
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "worker_script", str(fakes / "fake_seva_worker.py"))
    for mid in ("seva-1.1", "sd21-vae", "clip-vit-h-14"):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="explore", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["backend"]["id"] == "generative_scene"
    assert rep["coverage"]["by_splat"]["generative"] > 0.2
    assert {m["id"] for m in rep["models"]} >= {"seva-1.1", "vggt-1b"}
    req = json.loads((job.dir / "worker" / "fake_seva_worker_request.json").read_text())
    assert req["trajectory"] == "explore" and req["num_frames"] == 48
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["images"][0]["generated"] is False and mv["images"][0]["weight"] > 1
    assert all(v["generated"] for v in mv["images"][1:])
    assert any("GENERATIVE" in w for w in rep["warnings"])


def test_generative_request_falls_back_when_models_missing(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="orbit", layouts=["sbs"])
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


def test_dynamic_video_exports_4d_sequence(ctx, rtx4080, videos):
    job, rep, _ = run_job(ctx, rtx4080, videos["dynamic"], mode="fast", dynamic_fps=5, layouts=["sbs"],
                          export_sequence=True)
    assert rep["status"] == "succeeded", rep.get("error")
    seq = sorted((job.dir / "export" / "sequence").glob("frame_*.ply"))
    assert len(seq) >= 10
    dyn = [v for v in rep["outputs"]["vr180"]["videos"] if "dynamic" in Path(v["path"]).name][0]
    assert "stabilised" in dyn["method"] and dyn["sequence_dir"].endswith("sequence")


def test_generative_prefers_wan_when_installed(ctx, rtx4080, photo, monkeypatch):
    from conftest import REPO, install_fake_model
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "wan_script", str(fakes / "fake_seva_worker.py"))
    for mid in ("wan2.2-fun-5b-camera", "seva-1.1", "sd21-vae", "clip-vit-h-14"):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="arc", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert {m["id"] for m in rep["models"]} >= {"wan2.2-fun-5b-camera", "vggt-1b"}
    assert "seva-1.1" not in {m["id"] for m in rep["models"]}
    req = json.loads((job.dir / "worker" / "fake_seva_worker_request.json").read_text())
    assert req["trajectory"] == "arc" and req["frames"] == 25 and req["model_dir"].endswith("wan2.2-fun-5b-camera")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    # sharp fusion (default): the photo + a few key views, not every generated frame
    assert mv["assembly"] == "fusion" and mv["fuse_ref_side"] == 1024
    assert len(mv["images"]) == 17 and not mv["images"][0]["generated"]
    assert rep["backend"]["extra"]["assembly"] == "fusion"


def test_generative_trained_assembly_uses_every_frame(ctx, rtx4080, photo, monkeypatch):
    from conftest import REPO, install_fake_model
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "wan_script", str(fakes / "fake_seva_worker.py"))
    install_fake_model(ctx.models, "wan2.2-fun-5b-camera")
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", generative="arc", gen_assembly="train",
                          layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "worker" / "fake_seva_worker_request.json").read_text())
    assert req["frames"] == 49 and req["steps"] == 30
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["assembly"] == "train" and len(mv["images"]) == 50


def test_wan_out_of_memory_retries_in_a_fresh_process(ctx, rtx4080, photo, monkeypatch):
    from conftest import REPO, install_fake_model
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "wan_script", str(fakes / "fake_seva_worker.py"))
    monkeypatch.setenv("FAKE_WAN_OOM_ATTEMPTS", "2")
    install_fake_model(ctx.models, "wan2.2-fun-5b-camera")
    job, rep, log = run_job(ctx, rtx4080, photo, mode="fast", generative="arc", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "worker" / "fake_seva_worker_request.json").read_text())
    assert req["attempt"] == 2


def _fake_generative(monkeypatch):
    from conftest import REPO
    from twod2vr180.backends.generative import GenerativeSceneBackend
    from twod2vr180.backends.multiview import MultiViewBackend

    fakes = REPO / "tests" / "fakes"
    monkeypatch.setattr(MultiViewBackend, "worker_script", str(fakes / "fake_multiview_worker.py"))
    monkeypatch.setattr(GenerativeSceneBackend, "wan_script", str(fakes / "fake_seva_worker.py"))
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
    assert mv["assembly"] == "fusion" and len(mv["images"]) == n_views + 1
    assert {m["id"] for m in rep["models"]} >= set(QWEN_IDS)


def test_wan_360_capture_when_chosen(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS + ("wan2.2-fun-5b-camera",):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality", generative="capture", gen_engine="wan",
                          layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    req = json.loads((job.dir / "worker" / "fake_seva_worker_request.json").read_text())
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


@pytest.mark.parametrize("mode,n_cand", [("fast", 1), ("auto", 2), ("quality", 3)])
def test_qwen_three_views_go_to_trained_multiview(ctx, rtx4080, photo, monkeypatch, mode, n_cand):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode=mode, generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    gen = job.dir / "generated_views"
    enc = json.loads((gen / "stage_encode.json").read_text())
    assert [(v["azimuth"], v["elevation"]) for v in enc["views"]] == [(315, 0), (45, 0), (0, 30)]
    assert enc["angles_strength"] > 0 and enc["candidates"] == n_cand
    retry = 2 if n_cand > 1 else 0          # the fake scorer finds the first angle weak (Auto/Quality retry)
    assert len(list(gen.glob("view*.png"))) == 3 * n_cand + retry
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["assembly"] == "train" and len(mv["images"]) == 4      # the photo + exactly 3 views
    assert not mv["images"][0]["generated"]
    if n_cand > 1:
        assert [len(v["candidates"]) for v in mv["images"][1:]] == [n_cand + retry, n_cand, n_cand]
        assert all(v["target_deg"] in (45, 30) for v in mv["images"][1:])
    assert rep["backend"]["extra"]["assembly"] == "train"
    sheet = job.dir / "export" / "generated_views_sheet.jpg"
    assert sheet.exists() and rep["backend"]["extra"]["contact_sheet"].endswith("generated_views_sheet.jpg")


@pytest.mark.parametrize("with_8", [False, True])
def test_quality_uses_the_8_step_lightning_lora_when_installed(ctx, rtx4080, photo, monkeypatch, with_8):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS + (("qwen-edit-2511-lightning-8",) if with_8 else ()):
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    enc = json.loads((job.dir / "generated_views" / "stage_encode.json").read_text())
    assert enc["steps"] == (8 if with_8 else 4)
    assert ("8steps" in enc["lora_lightning"]) == with_8
    ids = {m["id"] for m in rep["models"]}
    assert ("qwen-edit-2511-lightning-8" in ids) == with_8


def test_multiview_backend_on_one_photo_generates_views_first(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", backend="multiview", generative="capture",
                          gen_assembly="fusion", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    assert rep["selection"]["backend"] == "generative_scene"
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["assembly"] == "train"


def test_multiview_backend_on_one_photo_without_generation_explains(ctx, rtx4080, photo):
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="fast", backend="multiview", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "failed" and "generative mode" in rep["error"]["message"]


def test_real3d_trains_at_photo_detail_with_low_sh(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, _ = run_job(ctx, rtx4080, photo, mode="quality", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert mv["max_side"] == 1600 and mv["sh_degree"] == 1



def test_weak_angle_gets_more_candidates(ctx, rtx4080, photo, monkeypatch):
    from conftest import install_fake_model

    _fake_generative(monkeypatch)
    for mid in QWEN_IDS:
        install_fake_model(ctx.models, mid)
    job, rep, log = run_job(ctx, rtx4080, photo, mode="auto", generative="tri", layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    gen = job.dir / "generated_views"
    req = json.loads((gen / "stage_generate.json").read_text())         # the last generate call (the retry)
    counts = {v["label"]: v.get("candidates") for v in req["views"]}
    assert counts["45° left"] == 4 and counts["45° right"] is None and counts["high angle"] is None
    assert len(list(gen.glob("view00*.png"))) == 4 and len(list(gen.glob("view01*.png"))) == 2


def test_rebuild_with_own_picks_marks_ai_views(ctx, rtx4080, photo, tmp_path, monkeypatch):
    from conftest import REPO
    from twod2vr180.backends.multiview import MultiViewBackend

    monkeypatch.setattr(MultiViewBackend, "worker_script", str(REPO / "tests" / "fakes" / "fake_multiview_worker.py"))
    import shutil

    picks = []
    for name in ("view00_c1.png", "view01.png"):
        shutil.copy(photo, tmp_path / name)
        picks.append(tmp_path / name)
    job, rep, _ = run_job(ctx, rtx4080, [photo] + picks, mode="quality", backend="multiview",
                          generated_inputs=[str(p) for p in picks], layouts=["sbs"], renderer="cpu")
    assert rep["status"] == "succeeded", rep.get("error")
    mv = json.loads((job.dir / "worker" / "fake_multiview_worker_request.json").read_text())
    assert [v["generated"] for v in mv["images"]] == [False, True, True]
    assert mv["images"][0]["weight"] >= 4 and mv["max_side"] == 1600 and mv["sh_degree"] == 1
