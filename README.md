# 2D2VR180 Autonomous Reconstruction

An autonomous build-and-integration project for Claude Code.

**Goal:** produce a Windows desktop application that accepts a photo or video and, using the user's local NVIDIA RTX 4080 16 GB GPU, can generate:

- an explorable 3D scene;
- Gaussian Splat output;
- optional mesh/OBJ output;
- stereoscopic VR180 SBS and/or Top/Bottom video;
- novel-view reconstruction/completion where technically possible.

The application must hide Python/CUDA/Conda/Git/Node/Visual Studio/PyTorch from the end user. Models and large runtimes are downloaded separately and are never committed to the Git repository.

## Important reality check

This repository is an **autonomous engineering blueprint**, not a claim that one current open-source model can perfectly reconstruct unseen geometry from a single photo/video. Single-view reconstruction is under-constrained. The project therefore uses a modular backend architecture and must preserve uncertainty instead of silently pretending generated geometry is observed geometry.

The first production target is a **working, reproducible Windows release for RTX 4080 16 GB**, not a research benchmark winner.

## Primary strategy

1. Photo:
   - fast single-view 3DGS backend;
   - optional single-image explorable-scene backend;
   - novel-view completion as an experimental stage.
2. Video:
   - frame selection;
   - VGGT/reconstruction backend;
   - LongSplat/3DGS where compatible;
   - mesh export;
   - optional generative novel-view repair.
3. VR180:
   - render two virtual eyes from the reconstructed scene;
   - fill missing/unsupported regions only when a validated completion backend is available;
   - encode Quest-friendly SBS/TB output.
4. Packaging:
   - Windows GUI;
   - self-contained application/runtime;
   - model manager;
   - no model weights in Git;
   - GitHub Release assets for installers and runtime bundles where size permits.

## Current state (2026-09-27)

The application skeleton, job system, VR180 renderer, exporters, model/runtime
managers, desktop GUI and Windows packaging are implemented and tested on CPU.
**No ML backend has yet been run on an RTX 4080**, so no release has been published.
See `docs/status/BOOTSTRAP.md`, `docs/status/EXECUTION_PLAN.md` and `docs/status/gates.json`.

### Quick start (source checkout)

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m twod2vr180.cli doctor          # GPU / driver / FFmpeg diagnostics
PYTHONPATH=src python -m twod2vr180.cli backends        # what can run here, and why not
PYTHONPATH=src python -m twod2vr180.cli runtimes install photo-cu128
PYTHONPATH=src python -m twod2vr180.cli models download moge-2-vitl-normal --accept-license
PYTHONPATH=src python -m twod2vr180.cli run photo.jpg   # -> jobs/<id>/export + run_report.json
PYTHONPATH=src python -m twod2vr180.cli vr180 scene.ply --layout sbs,tb
PYTHONPATH=src python -m twod2vr180.cli gui
```

Build instructions: `docs/BUILDING.md`.

### Layout

| Path | Content |
|---|---|
| `src/twod2vr180/` | application: hardware probe, media analysis, job system, selector, scene I/O, renderer, VR180, model & runtime managers, CLI, GUI |
| `src/twod2vr180/backends/` | adapters: `moge_rgbd`, `sharp`, `recon3d_video`, plus evaluated-but-unsupported research backends |
| `workers/` | scripts executed inside isolated backend runtimes (JSON-lines protocol) |
| `config/` | upstream commit lock, model manifest, runtime manifest, bundled-tool lock |
| `packaging/` | PyInstaller spec, Inno Setup installer |
| `scripts/` | bootstrap/build/test/package/release scripts and helpers |
| `tests/` | unit + integration tests (GPU-free; a test-only fake worker stands in for the model) |

## What Claude Code must NOT do

- Do not blindly vendor third-party repositories into this repository.
- Do not commit model weights.
- Do not commit CUDA/PyTorch/Conda runtimes.
- Do not redistribute weights whose license forbids redistribution.
- Do not replace a failed scientific backend with an unannounced fake/placeholder.
- Do not declare a pipeline production-ready without running its acceptance tests.
- Do not use `main` of an upstream repository at release time without recording the exact resolved commit.
- Do not put secrets in source, GitHub Actions logs, artifacts, or generated configuration.

See `CLAUDE.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`, `docs/REPOSITORY_MATRIX.md`, and `.claude/skills/`.
