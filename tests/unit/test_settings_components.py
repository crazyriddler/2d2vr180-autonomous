import json

from twod2vr.components import COMPONENTS, component_status, install_order
from twod2vr.settings import Settings


def test_settings_roundtrip_and_sanitize(tmp_path):
    p = tmp_path / "s.json"
    s = Settings(mode="fast", ipd_mm=58.0, video_fps=15.0, hf_token="x")
    s.save(p)
    r = Settings.load(p)
    assert r.mode == "fast" and r.ipd_mm == 58.0 and r.video_fps == 15.0
    # 1.x settings (VR180 options, presets) load without errors; nonsense is sanitized
    p.write_text(json.dumps({"mode": "bogus", "ipd_mm": 5, "vr180": True, "eye_resolution": 99999, "preset": "x"}))
    r = Settings.load(p)
    assert r.mode == "auto" and r.ipd_mm == 45.0
    p.write_text("{broken")
    assert Settings.load(p) == Settings()
    o = Settings(ipd_mm=60.0, real3d_fill=False, video_seconds=12.0, video_start_s=3.0).job_options()
    assert o.ipd_mm == 60.0 and not o.real3d_fill and o.video_seconds == 12.0 and o.video_start_s == 3.0


def test_component_catalogue(ctx):
    order = [c.id for c in install_order(["model-sapiens-pointmap"])]
    assert order == ["engine-photo", "model-sapiens-seg", "model-sapiens-pointmap"]
    for c in COMPONENTS:
        st = component_status(ctx, c)
        assert "installed" in st and "license" in st
    ids = {c.target for c in COMPONENTS}
    assert ids == {"photo-cu128", "gen-cu128", "sharp", "moge-3-vitl", "yunet-face", "sapiens2-seg-0.4b",
                   "sapiens2-pointmap-1b", "qwen-image-edit-2511-q5", "qwen-image-edit-2511-base",
                   "qwen-edit-2511-lightning"}
    # 'Install recommended' installs everything Real 3D uses
    assert all(c.recommended for c in COMPONENTS)


def test_models_of_older_versions_are_found_and_deleted(app_home):
    from twod2vr.models import ModelManager

    mm = ModelManager()
    old = mm.root / "vggt-1b"
    old.mkdir(parents=True)
    (old / "model.pt").write_bytes(b"x" * 1000)
    (mm.root / "sharp").mkdir()
    assert mm.unused() == [("vggt-1b", 1000)]
    assert mm.delete_unused() == 1000 and not old.exists() and (mm.root / "sharp").exists()
