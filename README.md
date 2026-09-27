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
