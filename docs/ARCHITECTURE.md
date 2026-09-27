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
