"""The real photo workers (moge_worker.py, sharp_depth_worker.py) run end to end on the CPU with small stand-ins
for MoGe-3, Depth Pro and SHARP (their weights need a GPU machine): the rc7 precision path (lens check,
several passes, second depth, normals, clean edges, whole-scene placement) and maximum detail. Needs torch."""

import json
import sys
import types
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
plyfile = pytest.importorskip("plyfile")
WORKERS = Path(__file__).resolve().parents[2] / "workers"
sys.path.insert(0, str(WORKERS))

SH_C0 = 0.28209479177387814
Gaussians3D = namedtuple("Gaussians3D", "mean_vectors singular_values quaternions colors opacities")


def truth_depth(h, w):
    """A wall 3 m away with fine ripples and a box 1.5 m away in front of it."""
    v, u = np.mgrid[0:h, 0:w] / max(h, w)
    d = 3.0 + 0.3 * u + 0.004 * np.sin(u * 300) * np.sin(v * 300)
    box = (u > 0.3) & (u < 0.6) & (v > 0.25) & (v < 0.6)
    return np.where(box, 1.5, d), box


class StubMoGe:
    """MoGe-3's interface: smooth metric depth (no ripples, soft edges), the true surface normals."""
    calls = []

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer(self, t, resolution_level=9, fov_x=None, use_fp16=False, refine_steps=3):
        import precision as pr

        StubMoGe.calls.append({"level": resolution_level, "fov_x": fov_x})
        h, w = t.shape[1:]
        fov = fov_x or 58.0
        f = w / 2 / np.tan(np.radians(fov) / 2)
        d, _ = truth_depth(h, w)
        img = t.permute(1, 2, 0).numpy()
        if img[0, -1, 0] > img[0, 0, 0]:      # the mirror image (bright border on the right): mirrored depth
            d = d[:, ::-1]
        n = pr.normals_from_depth(d, f)
        smooth = np.exp(pr.blur(np.log(d), 3.0, torch))
        K = torch.tensor([[f / w, 0, 0.5], [0, f / h, 0.5], [0, 0, 1]], dtype=torch.float32)
        pts = pr.rays(h, w, f) * smooth[..., None]
        return {"depth": torch.tensor(smooth, dtype=torch.float32), "mask": torch.ones((h, w), dtype=torch.bool),
                "intrinsics": K, "normal": torch.tensor(np.nan_to_num(n), dtype=torch.float32),
                "points": torch.tensor(pts, dtype=torch.float32)}


class StubSharp:
    """SHARP's interface: 2 layers of 768 x 768 splats at the given depth (else 2 m), pixel colours."""

    class Mono:
        def __init__(self, x):
            self.disparity = torch.full((1, 2) + tuple(x.shape[-2:]), 0.5)

    def load_state_dict(self, sd):
        return None

    def eval(self):
        return self

    def to(self, device):
        return self

    def monodepth_model(self, image):
        return StubSharp.Mono(image)

    def __call__(self, image, disparity_factor, depth=None):
        import torch.nn.functional as F

        g = 768
        z = torch.full((1, 1, g, g), 2.0) if depth is None else F.interpolate(depth, size=(g, g), mode="nearest")
        u = (torch.arange(g) + 0.5) * 2.0
        vv, uu = torch.meshgrid(u, u, indexing="ij")
        front = torch.stack([uu, vv, z[0, 0]], -1).reshape(-1, 3)
        back = torch.stack([uu, vv, z[0, 0] * 1.1], -1).reshape(-1, 3)
        col = F.interpolate(image, size=(g, g), mode="bilinear")[0].permute(1, 2, 0).reshape(-1, 3)
        n = 2 * g * g
        return Gaussians3D(torch.cat([front, back])[None], torch.full((1, n, 3), 0.003),
                           torch.tensor([1.0, 0, 0, 0]).repeat(n, 1)[None], torch.cat([col, col])[None],
                           torch.full((1, n), 0.9))


def unproject_gaussians(g, eye, K, size):
    m = g.mean_vectors[0].clone()
    z = m[:, 2]
    x = (m[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (m[:, 1] - K[1, 2]) / K[1, 1] * z
    return g._replace(mean_vectors=torch.stack([x, y, z], -1)[None])


def save_ply(g, f_px, hw, path):
    n = g.mean_vectors.shape[1]
    names = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3")
    v = np.zeros(n, [(k, "f4") for k in names])
    m = g.mean_vectors[0].float().numpy()
    v["x"], v["y"], v["z"] = m.T
    dc = (g.colors[0].float().numpy() - 0.5) / SH_C0
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc.T
    o = g.opacities[0].float().numpy().clip(1e-4, 1 - 1e-4)
    v["opacity"] = np.log(o / (1 - o))
    s = np.log(g.singular_values[0].float().numpy())
    v["scale_0"], v["scale_1"], v["scale_2"] = s.T
    v["rot_0"] = 1
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(path)


@pytest.fixture
def stubs(monkeypatch):
    moge = types.ModuleType("moge")
    moge_model = types.ModuleType("moge.model")
    v3 = types.ModuleType("moge.model.v3")
    v3.MoGeModel = StubMoGe
    flex = types.ModuleType("flex_gemm")
    flex.config = types.SimpleNamespace(AUTOTUNE_MODE="")
    sharp = types.ModuleType("sharp")
    sm = types.ModuleType("sharp.models")
    sm.PredictorParams = lambda: None
    sm.create_predictor = lambda p: StubSharp()
    su = types.ModuleType("sharp.utils")
    sg = types.ModuleType("sharp.utils.gaussians")
    sg.Gaussians3D, sg.unproject_gaussians, sg.save_ply = Gaussians3D, unproject_gaussians, save_ply
    cs = types.ModuleType("sharp.utils.color_space")
    cs.linearRGB2sRGB = lambda c: c
    su.color_space = cs
    for name, mod in {"moge": moge, "moge.model": moge_model, "moge.model.v3": v3, "flex_gemm": flex, "sharp": sharp,
                      "sharp.models": sm, "sharp.utils": su, "sharp.utils.gaussians": sg,
                      "sharp.utils.color_space": cs}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    StubMoGe.calls = []
    return StubMoGe


def events(out):
    return [json.loads(line[len("@@2D2VR "):]) for line in out.splitlines() if line.startswith("@@2D2VR ")]


def photo(path, w, h):
    from PIL import Image

    d, box = truth_depth(h, w)
    v, u = np.mgrid[0:h, 0:w]
    img = np.stack([120 + 60 * np.sin(u / 9.0), 110 + 50 * np.cos(v / 7.0), np.full((h, w), 140.0)], -1)
    img[box] = (230, 90, 60)
    img[:, :3] = (250, 250, 250)        # a bright left border: tells the stub MoGe which image is mirrored
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(path)
    return path


def test_precision_geometry_and_photo_end_to_end(stubs, tmp_path, capsys, monkeypatch):
    import moge_worker as mw
    import sharp_depth_worker as sd

    # Depth Pro: the true depth (sharp edges) at 1.1x the scale, a lens 2 % off the true one
    def dp_load(path, device, torch):
        return "dp"

    def dp_run(model, arr, device, torch):
        h, w = arr.shape[:2]
        d, _ = truth_depth(h, w)
        f = w / 2 / np.tan(np.radians(64.0) / 2)
        return f / (w * d * 1.1), 64.0 * 1.02

    monkeypatch.setattr(mw, "load_depth_pro", dp_load)
    monkeypatch.setattr(mw, "depth_pro_run", dp_run)
    src = photo(tmp_path / "p.png", 640, 480)
    (tmp_path / "dp.pt").write_bytes(b"x")
    req = {"model_path": "m.pt", "version": "v3", "max_side": 640, "allow_cpu": True,
           "images": [{"path": str(src), "out": str(tmp_path / "photo.npz"), "fov_x": 64.0}],
           "precision": {"passes": True, "depth_pro": str(tmp_path / "dp.pt"), "lens_check": True}}
    mw.main(req)
    res = [e for e in events(capsys.readouterr().out) if e["event"] == "result"][0]
    rep = res["precision"][0]
    assert rep["lens"]["from"] == "EXIF" and rep["passes"] == 3 and rep["depth_pro"]["used"]
    assert [c["fov_x"] for c in stubs.calls] == [None, 64.0, 64.0, 64.0]
    assert sorted(c["level"] for c in stubs.calls[1:]) == [7, 9, 9]
    geo = np.load(tmp_path / "photo.npz")
    assert {"normal", "depth_moge", "depth_pro", "points"} <= set(geo.files)
    # the joined depth keeps the box edge sharp (Depth Pro) where MoGe's was soft
    z, zm = geo["points"][..., 2], geo["depth_moge"]
    row = 240
    edge = slice(185, 200)                                          # the box's left edge at x = 192
    assert np.sum((z[row, edge] > 1.7) & (z[row, edge] < 2.8)) < np.sum((zm[row, edge] > 1.7) & (zm[row, edge] < 2.8))

    # SHARP with the precise depth: normals, clean edges, every seen splat placed, relief.png
    (tmp_path / "sharp.pt").write_bytes(b"")
    torch.save({}, tmp_path / "sharp.pt")
    out = tmp_path / "out"
    sreq = {"mode": "photo", "checkpoint": str(tmp_path / "sharp.pt"), "geometry": str(tmp_path / "photo.npz"),
            "output_dir": str(out), "face_model": None, "plain": False, "allow_cpu": True, "precision": True,
            "refine": True, "photo": str(src), "moge_path": "m.pt", "moge_version": "v3"}
    sd.main(sreq)
    ev = events(capsys.readouterr().out)
    res = [e for e in ev if e["event"] == "result"][0]
    p = res["precision"]
    assert p["normals"]["used"] and p["normals"]["rms_change"] > 0
    assert p["normals_fit"]["scene"]["ok"]
    assert p["edges"]["band_share"] > 0 and p["placed_share"] > 0.9
    assert "fit" not in p                                   # rc8: the fit runs in the renderer runtime
    assert Path(res["outputs"]["relief"]).exists()
    ply = plyfile.PlyData.read(res["outputs"]["locked"])["vertex"].data
    zs = np.asarray(ply["z"][: 768 * 768]).reshape(768, 768)
    # the splats on the wall follow the true ripples better than MoGe's smooth depth did
    zt, box = truth_depth(768, 768)
    sel = (~box) & (np.abs(zs - zt) < 0.3)
    sel[:, :20] = sel[:, -20:] = sel[:20] = sel[-20:] = False
    bumps_t = zt - np.exp(sd_blur(np.log(zt)))
    bumps_s = zs - np.exp(sd_blur(np.log(np.where(zs > 0, zs, 1))))
    assert np.corrcoef(bumps_t[sel], bumps_s[sel])[0, 1] > 0.3


def sd_blur(a):
    import precision as pr

    return pr.blur(a, 3.0, torch)


def test_maximum_detail_with_precision(stubs, tmp_path, capsys):
    import moge_worker as mw
    import sharp_depth_worker as sd

    src = photo(tmp_path / "big.png", 2200, 1600)
    req = {"model_path": "m.pt", "version": "v3", "max_side": 640, "allow_cpu": True,
           "images": [{"path": str(src), "out": str(tmp_path / "photo.npz")}],
           "precision": {"passes": False, "depth_pro": None, "lens_check": True}}
    mw.main(req)
    rep = [e for e in events(capsys.readouterr().out) if e["event"] == "result"][0]["precision"][0]
    assert rep["passes"] == 1 and rep["lens"]["exif_deg"] is None
    torch.save({}, tmp_path / "sharp.pt")
    sreq = {"mode": "photo", "checkpoint": str(tmp_path / "sharp.pt"), "geometry": str(tmp_path / "photo.npz"),
            "output_dir": str(tmp_path / "out"), "plain": False, "allow_cpu": True, "precision": True,
            "refine": False, "detail": True, "photo": str(src), "moge_path": "m.pt", "moge_version": "v3"}
    sd.main(sreq)
    res = [e for e in events(capsys.readouterr().out) if e["event"] == "result"][0]
    det = res["detail"]
    assert "error" not in det, det
    assert det["tiles"] == 4 and det["work_size"] == [2200, 1600] and det["splats"] > 2_000_000
    assert det["precision"]["normals"]["used"] and det["normals_fit"]["scene"]["ok"]
    assert res["outputs"]["locked"].endswith("locked_detail.ply") and res["outputs"]["standard"].endswith("locked.ply")
    assert res["size"] == [2200, 1600]


def test_the_fast_plan_skips_the_relief_from_normals(stubs, tmp_path, capsys):
    """rc9 Fast: one MoGe pass, no normals; clean edges and every splat placed still run."""
    import moge_worker as mw
    import sharp_depth_worker as sd

    src = photo(tmp_path / "p.png", 480, 360)
    mw.main({"model_path": "m.pt", "version": "v3", "max_side": 480, "allow_cpu": True,
             "images": [{"path": str(src), "out": str(tmp_path / "photo.npz"), "fov_x": 60.0}],
             "precision": {"passes": False, "depth_pro": None, "lens_check": True}})
    rep = [e for e in events(capsys.readouterr().out) if e["event"] == "result"][0]["precision"][0]
    assert rep["passes"] == 1
    torch.save({}, tmp_path / "sharp.pt")
    sd.main({"mode": "photo", "checkpoint": str(tmp_path / "sharp.pt"), "geometry": str(tmp_path / "photo.npz"),
             "output_dir": str(tmp_path / "out"), "plain": False, "allow_cpu": True, "precision": True,
             "normals": False, "photo": str(src)})
    res = [e for e in events(capsys.readouterr().out) if e["event"] == "result"][0]
    p = res["precision"]
    assert p["normals"] == {"used": False, "reason": "off in the Fast mode"} and "normals_fit" not in p
    assert p["edges"]["band_share"] > 0 and p["placed_share"] > 0.9 and "plain" not in res["outputs"]


def test_video_worker_cleans_edges_places_every_splat_and_adds_relief_in_quality(stubs, tmp_path, capsys):
    """rc10: the per-frame video steps of the balanced modes run in the real worker (stand-in models)."""
    import json as js

    import video_splat_worker as vw

    frames = [str(photo(tmp_path / f"f{i}.png", 320, 240)) for i in range(3)]
    torch.save({}, tmp_path / "sharp.pt")
    out = tmp_path / "out"
    vw.main({"frames": frames, "output_dir": str(out), "checkpoint": str(tmp_path / "sharp.pt"), "moge_path": "m.pt",
             "face_model": None, "human_seg": None, "human_pointmap": None, "fov_x": 60.0, "grid": [2, 4],
             "keyframe_interval": 2, "moge_max_side": 320, "refine_steps": 3, "people": True, "edges": True,
             "place_all": True, "normals": True, "allow_cpu": True})
    res = [e for e in events(capsys.readouterr().out) if e["event"] == "result"][0]
    assert res["frames"] == 3
    notes = [js.loads((out / "frames" / f"{i:06d}.bin.json").read_text()) for i in range(3)]
    assert all(n["relief"] for n in notes) and all(n["edges"] and n["edges"] > 0 for n in notes)
    assert {c["level"] for c in stubs.calls} == {9}
    # the box's edge: no splat left between the box (1.5 m) and the wall (3 m) in the front layer
    import splv_codec as codec

    meta = js.loads((out / "video_meta.json").read_text())
    arr = codec.decode_frame((out / "frames" / "000000.bin").read_bytes(), meta["count"], None)
    z = codec.dequantize(arr, meta["quant"])["means"][:, 2]
    front = z[: len(z) * 4 // 5]                          # the front layer comes first in the kept subset
    assert ((front > 1.8) & (front < 2.6)).mean() < 0.01

