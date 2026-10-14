import json
import re
from pathlib import Path

from twod2vr.backends.base import JobInput
from twod2vr.selector import select


def test_photo_goes_to_real3d_and_video_to_splat_video(ctx, rtx4080, tmp_path):
    for mode in ("auto", "quality", "fast"):
        assert select(JobInput(tmp_path / "a.jpg", "photo", tmp_path), rtx4080, ctx, mode).backend.id == "photo_locked"
        assert select(JobInput(tmp_path / "a.mp4", "video", tmp_path), rtx4080, ctx, mode).backend.id == "splat_video"


def test_missing_models_are_explained(ctx, rtx4080, tmp_path):
    ctx.models.delete("moge-3-vitl")
    sel = select(JobInput(tmp_path / "a.mp4", "video", tmp_path), rtx4080, ctx, "auto")
    assert sel.backend is None
    rej = {r["backend"]: r["reasons"] for r in sel.rejected}
    assert any("moge-3-vitl" in r for r in rej["splat_video"])
    rtx4080.gpus[0].total_mib = 6144
    sel = select(JobInput(tmp_path / "a.jpg", "photo", tmp_path), rtx4080, ctx, "auto")
    assert sel.backend is None and any("8 GB" in r for r in sel.rejected[0]["reasons"])


def test_runtime_manager_state(tmp_path):
    from twod2vr.runtimes import RuntimeManager, load_runtime_manifest

    specs = load_runtime_manifest()
    assert set(specs) == {"photo-cu128", "gen-cu128"}
    for s in specs.values():
        lines = RuntimeManager(tmp_path, specs).lock_lines(s.id)
        assert len(lines) > 20 and all("==" in p for p in lines), "runtimes must install an exact lock"
        torch_line = next(p for p in lines if p.startswith("torch=="))
        assert "+cu" in torch_line
        for a in s.archives_no_deps:   # commit archives, or prebuilt wheels pinned to a commit
            assert ("/archive/" in a["url"] and len(a["url"].rsplit("/", 1)[1]) == 44) or \
                (re.search(r"/raw/[0-9a-f]{40}/", a["url"]) and a["url"].endswith(".whl")), a["url"]
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
    from twod2vr.jobs import Job, JobOptions, JobRunner

    rtx4080.disk_free_mib = 10
    job = Job(photo, JobOptions())
    rep = JobRunner(ctx, rtx4080).run(job)
    assert rep["status"] == "failed" and rep["error"]["code"] == "low_disk"


def test_upstream_lock_complete():
    lock = json.loads((Path(__file__).resolve().parents[2] / "config" / "upstream-lock.json").read_text())
    for r in lock["repositories"]:
        assert len(r["commit"]) == 40 and r["code_license"] and r["category"] in lock["license_categories"]
        assert r["gpu_smoke_test"]


def test_copy_package_installs_subpackages_from_archive(tmp_path, monkeypatch):
    """stable-virtual-camera's packaging drops seva.modules; the installer copies the package tree."""
    import io
    import sys
    import urllib.request
    import zipfile

    from twod2vr.runtimes import RuntimeManager

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("repo-abc/seva/__init__.py", "")
        z.writestr("repo-abc/seva/modules/layers.py", "X = 1\n")
        z.writestr("repo-abc/demo.py", "")
        z.writestr("repo-abc/third_party/x.py", "")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp(buf.getvalue()))
    site = tmp_path / "site"
    site.mkdir()
    import subprocess

    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": str(site),
                                                                          "stderr": ""})())
    rm = RuntimeManager(root=tmp_path / "rt", specs={})
    rm._copy_package(sys.executable, {"url": "https://x/archive/abc.zip", "copy_package": "seva"}, lambda m: None, None)
    monkeypatch.setattr(subprocess, "run", real_run)
    assert (site / "seva" / "modules" / "layers.py").read_text() == "X = 1\n"
    assert not (site / "demo.py").exists() and not (site / "third_party").exists()
