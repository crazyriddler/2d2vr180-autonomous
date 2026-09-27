import json
from pathlib import Path

from conftest import install_fake_model

from twod2vr180.backends.base import JobInput
from twod2vr180.selector import select


def test_sharp_preferred_when_installed(ctx, rtx4080, tmp_path):
    install_fake_model(ctx.models, "sharp")
    inp = JobInput(tmp_path / "a.jpg", "photo", tmp_path)
    assert select(inp, rtx4080, ctx, "quality").backend.id == "sharp"
    assert select(inp, rtx4080, ctx, "fast").backend.id == "moge_rgbd"


def test_fallback_is_explained(ctx, rtx4080, tmp_path):
    rtx4080.gpus[0].total_mib = 8192  # too small for recon3d
    inp = JobInput(tmp_path / "a.mp4", "video", tmp_path, video_kind="moving_camera")
    sel = select(inp, rtx4080, ctx, "auto")
    assert sel.backend.id == "moge_rgbd" and sel.effective_kind == "video:static_scene"
    rej = {r["backend"]: r["reasons"] for r in sel.rejected}
    assert any("12 GB" in r for r in rej["recon3d_video"])
    assert "longsplat" in rej  # unsupported research backends are listed with reasons


def test_runtime_manager_state(tmp_path):
    from twod2vr180.runtimes import RuntimeManager, load_runtime_manifest

    specs = load_runtime_manifest()
    assert {"photo-cu128", "recon3d-cu124"} <= set(specs)
    for s in specs.values():
        lines = RuntimeManager(tmp_path, specs).lock_lines(s.id)
        assert len(lines) > 20 and all("==" in p for p in lines), "runtimes must install an exact lock"
        torch_line = next(p for p in lines if p.startswith("torch=="))
        assert "+cu" in torch_line
        assert all("/archive/" in a["url"] and len(a["url"].rsplit("/", 1)[1]) == 44 for a in s.archives_no_deps)
    rm = RuntimeManager(tmp_path, specs)
    assert not rm.is_installed("photo-cu128")
    py = rm.python("photo-cu128")
    py.parent.mkdir(parents=True)
    py.write_text("")
    (rm.env_dir("photo-cu128") / "installed.json").write_text(
        json.dumps({"spec_fingerprint": rm._fingerprint("photo-cu128")}))
    assert rm.is_installed("photo-cu128")
    specs["photo-cu128"].local_versions["torch"] = "+cu999"  # manifest change invalidates the install
    assert not rm.is_installed("photo-cu128")


def test_low_disk_job_failure(ctx, rtx4080, photo):
    from twod2vr180.jobs import Job, JobOptions, JobRunner

    rtx4080.disk_free_mib = 10
    job = Job(photo, JobOptions())
    rep = JobRunner(ctx, rtx4080).run(job)
    assert rep["status"] == "failed" and rep["error"]["code"] == "low_disk"


def test_upstream_lock_complete():
    lock = json.loads((Path(__file__).resolve().parents[2] / "config" / "upstream-lock.json").read_text())
    for r in lock["repositories"]:
        assert len(r["commit"]) == 40 and r["code_license"] and r["category"] in lock["license_categories"]
        assert r["gpu_smoke_test"]
