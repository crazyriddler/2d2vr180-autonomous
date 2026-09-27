import json

from twod2vr180.components import COMPONENTS, component_status, install_order
from twod2vr180.settings import Settings


def test_settings_roundtrip_and_sanitize(tmp_path):
    p = tmp_path / "s.json"
    s = Settings(mode="fast", eye_separation_mm=70, layout_tb=False, hf_token="x")
    s.save(p)
    r = Settings.load(p)
    assert r.mode == "fast" and r.eye_separation_mm == 70 and not r.layout_tb
    p.write_text(json.dumps({"mode": "bogus", "eye_resolution": 99999, "unknown_key": 1}))
    r = Settings.load(p)
    assert r.mode == "auto" and r.eye_resolution == 4096
    p.write_text("{broken")
    assert Settings.load(p) == Settings()
    o = Settings(layout_sbs=False, layout_tb=True, eye_separation_mm=60).job_options()
    assert o.layouts == ["tb"] and abs(o.eye_separation_m - 0.06) < 1e-9


def test_component_catalogue(ctx):
    order = [c.id for c in install_order(["model-sharp", "model-moge-l"])]
    assert order == ["engine-photo", "model-sharp", "model-moge-l"]
    for c in COMPONENTS:
        st = component_status(ctx, c)
        assert "installed" in st and "license" in st
    ids = {c.target for c in COMPONENTS}
    assert {"photo-cu128", "recon3d-cu124", "sharp", "depth-anything-v2-small"} <= ids
