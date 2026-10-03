"""Command-line interface: ``2d2vr180 <command>``."""

from __future__ import annotations

import argparse
import urllib.parse
import json
import sys
from pathlib import Path

from . import __version__


def _ctx(profile: str = "personal_research"):
    from .backends.base import BackendContext
    from .models import ModelManager
    from .runtimes import RuntimeManager

    return BackendContext(ModelManager(), RuntimeManager(), profile)


def cmd_doctor(a) -> int:
    from .hardware import format_report, probe
    from .runtimes import find_uv

    hw = probe()
    if a.json:
        d = hw.to_dict()
        d["uv"] = find_uv()
        print(json.dumps(d, indent=2))
    else:
        print(f"2D2VR180 {__version__}")
        print(format_report(hw))
        print(f"uv:        {find_uv() or 'NOT FOUND (runtime installation disabled)'}")
    return 0 if hw.gpus else 3


def cmd_backends(a) -> int:
    from .backends import all_backends
    from .hardware import probe

    ctx = _ctx(a.license_profile)
    hw = probe()
    rows = []
    for b in all_backends():
        av = b.availability(ctx, hw, {"mode": "auto"})
        rows.append({**b.describe(), "available": av.ok, "reasons": av.reasons})
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            mark = "OK " if r["available"] else "-- "
            print(f"{mark}{r['id']:<16} [{r['maturity']}] {r['name']}")
            for why in r["reasons"]:
                print(f"      {why}")
    return 0


def cmd_models(a) -> int:
    from .models import LicenseNotAccepted, ModelError, ModelManager

    mm = ModelManager()
    if a.action == "list":
        for mid in mm.entries:
            s = mm.status(mid)
            size = f"{s['size_bytes'] / 2**30:.2f} GiB" if s["size_bytes"] else "size unknown"
            mark = "[x]" if s["installed"] else ("[~]" if mm.entries[mid].extra.get("fetched_by_upstream") else "[ ]")
            print(f"{mark} {mid:<22} {size:<14} {s['license']}")
        print("[~] = fetched by its backend runtime on first use")
        return 0
    if not a.model:
        print("model id required", file=sys.stderr)
        return 2
    if a.action == "download":
        e = mm.entries.get(a.model)
        if e is None:
            print(f"unknown model {a.model}", file=sys.stderr)
            return 2
        if e.extra.get("fetched_by_upstream"):
            print(f"{a.model} is fetched by its backend runtime on first use; nothing to download here.")
            return 0
        print(f"Model:   {e.display_name}\nSource:  {e.source}\nLicense: {e.license}\n         {e.license_url}\n"
              f"Commercial use: {e.commercial_use}   Redistributable: {e.redistributable}")
        if not mm.license_accepted(a.model):
            if not a.accept_license:
                print("Re-run with --accept-license after reading the license above.", file=sys.stderr)
                return 4
            mm.accept_license(a.model)

        def prog(done, total):
            t = f"/{total / 2**20:.0f}" if total else ""
            print(f"\r  {done / 2**20:.0f}{t} MiB", end="", flush=True)

        try:
            mm.download(a.model, progress=prog)
        except (ModelError, LicenseNotAccepted) as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 1
        print("\nOK")
        return 0
    if a.action == "verify":
        probs = mm.verify(a.model)
        print("OK" if not probs else "\n".join(probs))
        return 0 if not probs else 1
    if a.action == "delete":
        mm.delete(a.model)
        print("deleted")
        return 0
    return 2


def cmd_runtimes(a) -> int:
    from .runtimes import RuntimeInstallError, RuntimeManager

    rm = RuntimeManager()
    if a.action == "list":
        for rid in rm.specs:
            s = rm.status(rid)
            print(f"{'[x]' if s['installed'] else '[ ]'} {rid:<16} ~{s['approx_size_gb']} GB  {s['description']}")
            print(f"      manifest status: {s['manifest_status']}")
        return 0
    if a.action == "install":
        try:
            rm.install(a.runtime)
        except (RuntimeInstallError, KeyError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0
    if a.action == "remove":
        rm.remove(a.runtime)
        return 0
    return 2


def cmd_run(a) -> int:
    from .jobs import Job, JobOptions, JobRunner

    opts = JobOptions(mode=a.mode, backend=a.backend, vr180=not a.no_vr180,
                      layouts=[x for x in a.layout.split(",") if x], projection=a.projection,
                      eye_resolution=a.eye_resolution, license_profile=a.license_profile,
                      output_dir=a.out, allow_cpu=a.allow_cpu, renderer=a.renderer,
                      fill_holes=not a.no_fill_holes, video_eye_resolution=a.video_eye_resolution,
                      ai_hole_fill=not a.no_ai_fill, generative=a.generative, gen_assembly=a.assembly, gen_engine=a.engine,
                      video_mode=a.video_mode,
                      export_sequence=a.export_sequence)
    job = Job([Path(p) for p in a.input] if len(a.input) > 1 else Path(a.input[0]), opts)
    runner = JobRunner(_ctx(a.license_profile))

    def on_event(ev):
        if ev["event"] == "progress":
            print(f"[{ev['value'] * 100:5.1f}%] {ev.get('message', '')}", flush=True)
        elif ev["event"] == "log" and a.verbose:
            print(f"        {ev.get('message', '')}", flush=True)

    rep = runner.run(job, on_event)
    print(f"\nstatus: {rep['status']}   report: {job.dir / 'run_report.json'}")
    if a.summary_json:
        out = rep.get("outputs") or {}
        summary = {"status": rep["status"], "error": rep.get("error"), "backend": (rep.get("backend") or {}).get("id"),
                   "worker_env": (rep.get("backend") or {}).get("worker_env"), "validation": rep.get("validation"),
                   "coverage": rep.get("coverage"), "gaussians": out.get("gaussians"),
                   "outputs": {k: out.get(k) for k in ("scene_ply", "splat", "obj")},
                   "vr180": [s["metadata"].get("coverage") for s in (out.get("vr180") or {}).get("stills", [])],
                   "videos": [{k: v.get(k) for k in ("path", "frames", "spherical_metadata")}
                              for v in (out.get("vr180") or {}).get("videos", [])],
                   "timings_s": rep.get("timings_s"), "warnings": rep.get("warnings"), "report": str(job.dir)}
        Path(a.summary_json).write_text(json.dumps(summary, indent=2, default=str))
    for w in rep.get("warnings", []):
        print(f"WARNING: {w}")
    if rep.get("error"):
        print(f"ERROR [{rep['error']['code']}]: {rep['error']['message']}", file=sys.stderr)
        return 1
    return 0


def cmd_vr180(a) -> int:
    import numpy as np

    from .scene import load_scene
    from .vr180 import StereoOptions, render_stereo, save_stereo_image

    scene = load_scene(Path(a.scene))
    if a.metric:
        scene.metric_scale = True
    out = Path(a.out or Path(a.scene).parent)
    for layout in a.layout.split(","):
        so = StereoOptions(layout=layout, projection=a.projection, eye_resolution=a.eye_resolution,
                           eye_separation_m=a.eye_separation)
        fr = render_stereo(scene, so, head_c2w=scene.cameras[0].c2w if scene.cameras else np.eye(4))
        saved = save_stereo_image(fr, so, out, Path(a.scene).stem)
        print(saved["image"])
        if fr.metadata.get("honesty_note"):
            print("NOTE:", fr.metadata["honesty_note"])
    return 0


def cmd_view_vr(a) -> int:
    """Serve a splat to the bundled WebXR viewer on 127.0.0.1 and open it."""
    import time
    import urllib.request
    from .scene import load_scene
    from .vr_server import get_server, view_params

    ply = Path(a.scene)
    scene = load_scene(ply)
    depth, xf = view_params(scene)
    url = get_server().share(ply, depth, ply.stem, xf)
    print(url, flush=True)
    if a.check:  # self-test: viewer page, modules and the scene itself are served
        base = url.split("/viewer.html")[0]
        scene_path = urllib.parse.unquote(url.split("scene=")[1].split("&")[0])
        for part in ("/viewer.html", "/three.module.js", "/gaussian-splats-3d.module.js", scene_path):
            with urllib.request.urlopen(base + part, timeout=10) as r:
                if r.status != 200:
                    return 1
        print("VR viewer OK")
        return 0
    if not a.no_browser:
        from .vr_server import open_in_browser

        print(f"opened in: {open_in_browser(url)}")
    print("Serving on this computer only. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


def cmd_gui(a) -> int:
    from .gui.app import main as gui_main

    return gui_main([a.open] if getattr(a, "open", None) else [])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="2d2vr180", description="Local photo/video → 3D / VR180")
    p.add_argument("--version", action="version", version=f"2D2VR180 {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="hardware / driver / FFmpeg diagnostics")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_doctor)

    b = sub.add_parser("backends", help="list backends and why they are (un)available")
    b.add_argument("--json", action="store_true")
    b.add_argument("--license-profile", default="personal_research", choices=["personal_research", "commercial"])
    b.set_defaults(fn=cmd_backends)

    m = sub.add_parser("models", help="model manager")
    m.add_argument("action", choices=["list", "download", "verify", "delete"])
    m.add_argument("model", nargs="?")
    m.add_argument("--accept-license", action="store_true")
    m.set_defaults(fn=cmd_models)

    r = sub.add_parser("runtimes", help="isolated backend runtimes")
    r.add_argument("action", choices=["list", "install", "remove"])
    r.add_argument("runtime", nargs="?")
    r.set_defaults(fn=cmd_runtimes)

    j = sub.add_parser("run", help="reconstruct a photo or video")
    j.add_argument("input", nargs="+", help="a photo or video; several photos (or a folder) make one multi-view scene")
    j.add_argument("--mode", default="auto", choices=["auto", "quality", "fast"])
    j.add_argument("--backend")
    j.add_argument("--no-vr180", action="store_true")
    j.add_argument("--layout", default="sbs,tb")
    j.add_argument("--projection", default="equirect180", choices=["equirect180", "flat"])
    j.add_argument("--eye-resolution", type=int, default=2048)
    j.add_argument("--license-profile", default="personal_research", choices=["personal_research", "commercial"])
    j.add_argument("--out", help="also copy outputs to this directory")
    j.add_argument("--video-eye-resolution", type=int, default=1280)
    j.add_argument("--allow-cpu", action="store_true", help="run cpu-capable backends without a GPU (slow)")
    j.add_argument("--renderer", default="auto", choices=["auto", "gpu", "cpu"])
    j.add_argument("--no-fill-holes", action="store_true")
    j.add_argument("--no-ai-fill", action="store_true", help="do not use LaMa for VR180 holes")
    j.add_argument("--generative", default="off", choices=["off", "tri", "capture", "arc", "orbit", "explore", "spiral"],
                   help="photos: invent unseen views (Wan 2.2 / Stable Virtual Camera) and build a full 3D scene")
    j.add_argument("--engine", default="auto", choices=["auto", "qwen", "wan", "seva"],
                   help="generative 3D: model that invents the other views (auto: Qwen, then Wan, then SEVA)")
    j.add_argument("--assembly", default="fusion", choices=["fusion", "train"],
                   help="generative 3D: sharp fusion of per-view MoGe-2 geometry (default) or one trained splat")
    j.add_argument("--export-sequence", action="store_true", help="fixed-camera video: one .ply per frame")
    j.add_argument("--video-mode", default="auto", choices=["auto", "multiview", "per_frame", "best_frame"])
    j.add_argument("--summary-json", help="write a compact result summary to this file")
    j.add_argument("-v", "--verbose", action="store_true")
    j.set_defaults(fn=cmd_run)

    v = sub.add_parser("vr180", help="render VR180/stereo stills from an existing .ply/.splat")
    v.add_argument("scene")
    v.add_argument("--layout", default="sbs")
    v.add_argument("--projection", default="equirect180", choices=["equirect180", "flat"])
    v.add_argument("--eye-resolution", type=int, default=2048)
    v.add_argument("--eye-separation", type=float, default=0.064)
    v.add_argument("--metric", action="store_true", help="scene units are metres")
    v.add_argument("--out")
    v.set_defaults(fn=cmd_vr180)

    x = sub.add_parser("view-vr", help="open a .ply/.splat in the WebXR viewer (PC-VR headsets via Link/SteamVR)")
    x.add_argument("scene")
    x.add_argument("--no-browser", action="store_true")
    x.add_argument("--check", action="store_true", help="self-test the viewer server and exit")
    x.set_defaults(fn=cmd_view_vr)

    g = sub.add_parser("gui", help="launch the desktop application")
    g.add_argument("open", nargs="?")
    g.set_defaults(fn=cmd_gui)
    return p


def _safe_console() -> None:
    """Windows consoles default to a legacy code page (cp1252/cp850) that cannot
    encode characters such as '→' or '°'; never let console output crash a job."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
