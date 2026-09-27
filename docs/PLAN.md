# Master Engineering Plan

## Phase 0 — Repository reconnaissance

Goal: establish the real current state of every candidate project.

Candidates:

- https://github.com/jaskirat1616/Splatline
- https://github.com/jashshah999/recon3d
- https://github.com/facebookresearch/vggt
- https://github.com/Wang-pengfei/One2Scene
- https://github.com/apple/ml-sharp
- https://github.com/NVlabs/LongSplat
- https://github.com/DongLiangSXU/DepthSplat
- https://github.com/GVCLab/GSFixer
- https://github.com/GSFix3D/GSFix3D
- https://github.com/nerfstudio-project/gsplat

The exact URLs must be verified by the agent before use. Do not assume a repository still exists under the same owner/path.

Deliverable:
`config/upstream-lock.json` containing exact resolved commits, licenses, model licenses, build notes and status.

## Phase 1 — Prove the local hardware path

Create `tools/diagnostics` that reports:
- Windows version;
- NVIDIA driver;
- CUDA runtime visible to PyTorch/backend;
- RTX model;
- total/free VRAM;
- CPU/RAM;
- disk space;
- FFmpeg availability.

Create a minimal GPU smoke test.

Target: RTX 4080 16 GB.

## Phase 2 — Backend adapters

Implement a common interface:

`Backend.prepare()`
`Backend.can_run(input, hardware)`
`Backend.estimate(input, hardware)`
`Backend.run(input, options, progress_callback)`
`Backend.export(result, destination)`
`Backend.cleanup()`

Each backend runs in its own isolated environment/runtime where required.

## Phase 3 — Photo pipeline

First production candidate:
- SHARP or another verified single-image feed-forward 3DGS backend if license permits the user's intended use.
- Alternative open backend if licensing or Windows compatibility blocks SHARP.

Second candidate:
- One2Scene for explorable single-image scenes.

The agent must benchmark:
- fidelity to source;
- camera-motion stability;
- VRAM;
- runtime;
- output quality.

## Phase 4 — Video pipeline

First production candidate:
- recon3d as integration reference;
- VGGT for camera/depth;
- gsplat for splatting.

Then evaluate:
- LongSplat;
- DepthSplat;
- factor-graph refinement.

Important: a camera-fixed video is a special case. If the scene is static, many frames are redundant. If people/animals move, temporal processing can help observe changing surfaces but may violate static-scene assumptions. The pipeline selector must distinguish these cases where practical.

## Phase 5 — Human/animal/object specialization

Do not force one reconstruction algorithm to handle everything.

Implement capability tags:
- `scene_static`
- `object_centric`
- `human`
- `animal`
- `dynamic`
- `single_view`
- `multi_view`
- `novel_view_completion`

Add specialist backends only after license and local-hardware validation.

## Phase 6 — Novel-view completion

Candidate:
- GSFixer
- GSFix3D
- One2Scene's novel-view generator
- other validated 2026 projects discovered during reconnaissance

This is experimental until:
- it preserves source-view identity;
- does not visibly change observed pixels without permission;
- produces temporally stable output;
- runs within RTX 4080 constraints or has a documented fallback.

## Phase 7 — VR180

Do not convert a 2D frame to SBS merely by shifting pixels.

Preferred pipeline:

3D representation
→ virtual left/right cameras
→ render
→ repair missing/disoccluded areas
→ equirectangular/hemispherical projection
→ SBS/TB encode.

The first supported VR180 mode may use a constrained field of view if full hemispherical completion is not reliable. The UI must report the actual FOV and generated coverage.

## Phase 8 — Desktop application

Recommended architecture:
- Python backend is allowed internally.
- GUI should be a compiled desktop shell.
- IPC between GUI and worker process.
- worker process owns ML runtime.
- FFmpeg handles media I/O.
- job queue with cancel/resume.
- logs and progress.

Candidate GUI technologies:
- PySide6 packaged with PyInstaller/Nuitka;
- Tauri/Electron only if they materially simplify packaging.

Do not choose a technology based on fashion. Benchmark build size, startup time, GPU process management and installer reliability.

## Phase 9 — Runtime/model manager

Models are downloaded on demand.

Every model entry needs:
- name;
- URL;
- source;
- exact revision;
- license;
- SHA256;
- size;
- required VRAM;
- supported backends;
- redistribution permission;
- local path.

Downloads must be resumable and atomic:
`.part` → verify → rename.

## Phase 10 — Packaging

Produce:
- portable ZIP;
- installer EXE;
- optional offline runtime bundle where licensing allows.

Do not bundle restricted weights.

For every release:
- `checksums.sha256`
- `release-manifest.json`
- human-readable release notes
- hardware support matrix.

## Phase 11 — Acceptance

Mandatory test inputs:
- simple indoor scene;
- object;
- human;
- animal if an openly licensed test asset is available;
- static camera video;
- moving-subject fixed-camera video.

Tests must include:
- launch;
- model download;
- inference;
- cancellation;
- low-disk handling;
- insufficient VRAM handling;
- missing model handling;
- output validation;
- corrupted input handling.

## Phase 12 — Release

Only after acceptance:
- tag `vX.Y.Z`;
- build release assets;
- upload to GitHub Release;
- generate release notes;
- attach checksums;
- record exact upstream commits.

## Long-term roadmap

V1:
- reliable photo/video → 3DGS;
- viewer;
- OBJ;
- basic VR180.

V2:
- One2Scene/single-image exploration;
- better novel-view completion;
- improved fixed-camera video.

V3:
- specialist human/animal pipelines;
- stronger temporal consistency;
- quality presets;
- automated backend selection.
