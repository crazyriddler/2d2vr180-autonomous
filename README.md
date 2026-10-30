# 2D2VR

A Windows desktop application that turns, on your own NVIDIA GPU (target: RTX 4080 16 GB):

- **a photo into a Real 3D Gaussian splat**: Apple SHARP's 3D Gaussians aligned to the photo's measured
  metric depth (MoGe-3) at its real lens, people with their real 3D shape (Sapiens2), the real-world size set
  from their eyes; three quality modes (Fast, Auto, Quality) that each add visible precision;
- **a video into a splat video** (`.splv`): every frame made the same way, one file with the sound, played
  smoothly in the app's 3D viewer and in the browser VR player (WebXR), where the controller buttons resize
  the scene.

Nothing you process leaves the computer. The end user never installs Python, CUDA, Git or anything else:
the app downloads its engines and models once, with their licences shown first.

2.0 is a complete reform of 1.x ("2D2VR180"): the VR180 stereo output, the 4D video method and the models they
needed were removed. See `docs/RELEASE_NOTES.md` and `docs/USER_GUIDE.md`.

## Quick start (source checkout)

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m twod2vr.cli doctor          # GPU / driver / FFmpeg diagnostics
PYTHONPATH=src python -m twod2vr.cli backends        # what can run here, and why not
PYTHONPATH=src python -m twod2vr.cli runtimes install photo-cu128
PYTHONPATH=src python -m twod2vr.cli models download moge-3-vitl --accept-license
PYTHONPATH=src python -m twod2vr.cli models download sharp --accept-license
PYTHONPATH=src python -m twod2vr.cli run photo.jpg   # -> jobs/<id>/export + run_report.json
PYTHONPATH=src python -m twod2vr.cli run clip.mp4    # -> jobs/<id>/export/scene.splv
PYTHONPATH=src python -m twod2vr.cli view-vr jobs/<id>/export/scene.splv
PYTHONPATH=src python -m twod2vr.cli gui
```

Build instructions: `docs/BUILDING.md`. Architecture: `docs/ARCHITECTURE.md`.

## Layout

| Path | Content |
|---|---|
| `src/twod2vr/` | application: hardware probe, media, job system, selector, scene I/O, splat video I/O, model & runtime managers, VR player server, CLI, GUI |
| `src/twod2vr/backends/` | `photo_locked` (Real 3D for photos, its quality plans in `PHOTO_PLANS`) and `splat_video` (videos) |
| `workers/` | scripts executed inside the isolated GPU runtimes (JSON-lines protocol); `splv_codec.py` is the splat video format |
| `assets/webxr/` | the browser player (`player.html`, `player.js`) for `.splat` and `.splv` |
| `config/` | upstream commit lock, model manifest, runtime manifest, bundled-tool lock |
| `packaging/` | PyInstaller spec, Inno Setup installer |
| `scripts/` | bootstrap/build/test/package/release scripts and helpers |
| `tests/` | unit + integration tests (GPU-free; test-only fake workers stand in for the models; the player runs in headless Chromium) |

## Rules for contributors (and Claude Code)

- Do not commit model weights or runtimes; do not vendor third-party repositories without a licence note.
- Never present AI-generated geometry as observed: every splat carries its provenance.
- Do not declare a pipeline ready without its tests; record exact upstream commits.
- No secrets in source, logs or artifacts.

See `CLAUDE.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md` and `.claude/skills/`.
