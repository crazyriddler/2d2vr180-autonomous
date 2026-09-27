# Architecture

```text
                    ┌─────────────────────┐
                    │     2D2VR180 GUI    │
                    └──────────┬──────────┘
                               │ IPC
                    ┌──────────▼──────────┐
                    │     Job Manager     │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
       Input Analyzer     Hardware Probe    Model Manager
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Pipeline Selector   │
                    └──────────┬──────────┘
                               │
       ┌───────────────┬───────┼───────────────┬──────────────┐
       ▼               ▼       ▼               ▼              ▼
     PHOTO           VIDEO    HUMAN          REPAIR          VR
     backend         backend  backend        backend         renderer
       │               │       │               │              │
       └───────────────┴───────┴───────────────┴──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Unified Scene Model │
                    └──────┬───────┬─────┘
                           │       │
                         3DGS     Mesh
                           │       │
                           └───┬───┘
                               ▼
                         View Renderer
                               │
                        ┌──────┴──────┐
                        ▼             ▼
                     Normal        VR180
                     render       SBS / TB
```

## Design rule

The GUI never imports research repositories directly.

Each upstream project gets an adapter.

This protects the application from upstream API churn.

## Process isolation

Research ML backends may have conflicting Python/Torch/CUDA dependencies. Do not try to force every backend into one Python environment.

Preferred:
- one worker/runtime per compatible dependency family;
- process isolation;
- explicit protocol;
- temporary working directories;
- cleanup after job.

## Unified scene contract

Each pipeline returns:

```json
{
  "scene_id": "...",
  "source": {
    "path": "...",
    "type": "photo|video"
  },
  "representation": {
    "type": "gaussian_splat|mesh|hybrid",
    "path": "..."
  },
  "camera": {
    "intrinsics": "...",
    "poses": "..."
  },
  "coverage": {
    "observed": "...",
    "inferred": "...",
    "generative": "..."
  },
  "quality": {
    "score": null,
    "warnings": []
  },
  "backend": {
    "name": "...",
    "commit": "...",
    "model_revision": "..."
  }
}
```

## GPU memory policy

RTX 4080 has 16 GB VRAM.

The runtime must:
- detect available VRAM;
- choose conservative defaults;
- process videos in chunks;
- release CUDA memory between stages;
- avoid running two large models simultaneously;
- support CPU offload when a backend provides it;
- fail with actionable diagnostics rather than OOM loops.

## Windows packaging

The final user-facing package should contain:
- GUI;
- worker launcher;
- required native libraries;
- FFmpeg;
- runtime files required by the selected backends.

Large ML models are downloaded by the model manager.

## Decisions taken in the first implementation pass (2026-09-27)

- **GUI: PySide6 + PyInstaller (one folder) + Inno Setup.** The application process only
  needs numpy/Pillow/Qt (≈ 300 MB frozen with the bundled tools); ML code never loads in it.
  Tauri/Electron would add a Node toolchain without simplifying GPU process management.
- **Runtimes: bundled `uv`.** Each backend family gets a separate environment under
  `%LOCALAPPDATA%\2D2VR180\runtimes`, installed from pinned wheels (PyTorch CUDA wheels
  carry the CUDA runtime). Git dependencies are installed from GitHub commit archives, so
  neither Git nor a compiler is needed. Backends requiring compiled CUDA extensions are
  excluded until prebuilt wheels exist.
- **Worker protocol:** `worker.py request.json`, JSON lines prefixed `@@2D2VR180 ` on stdout
  (`progress`, `log`, `env`, `result`, `error{code}`); cancellation kills the process tree.
- **Scene convention:** OpenCV camera frame of the reference camera, metres when metric.
  Per-splat provenance (observed / inferred / generative / unknown) is kept in a sidecar so the
  exported PLY stays compatible with third-party viewers.
- **Reference renderer:** numpy z-buffered isotropic splats for preview, validation and VR180.
  It is an approximation of 3DGS alpha compositing and is labelled as such in reports; a gsplat
  path inside runtimes is the planned quality renderer.
- **License profiles:** "personal/research" (default) and "commercial" — the latter disables
  every backend whose weights are not licensed for commercial use.
