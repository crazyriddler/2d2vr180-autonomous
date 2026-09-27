# Agent Organization

Claude Code should delegate by responsibility, not by repository name.

## Lead / Orchestrator

Owns:
- roadmap;
- architecture;
- release decision;
- integration contracts;
- license gate.

## Upstream Research Agent

Owns:
- repository reconnaissance;
- current commit;
- installation;
- model URLs;
- license;
- Windows/CUDA notes;
- benchmark notes.

## Geometry Agent

Owns:
- VGGT;
- depth;
- camera poses;
- scene coordinate normalization;
- 3DGS adapters.

## Photo Agent

Owns:
- single-image backends;
- One2Scene;
- SHARP and alternatives;
- source-view fidelity.

## Video Agent

Owns:
- frame extraction;
- static camera detection;
- chunking;
- temporal consistency;
- LongSplat/DepthSplat/recon3d integration.

## Generative Repair Agent

Owns:
- GSFixer;
- GSFix3D;
- unseen/disoccluded regions;
- temporal consistency;
- safeguards against hallucinating observed pixels.

## VR Agent

Owns:
- stereo camera model;
- equirectangular projection;
- SBS/TB;
- Quest-compatible encoding;
- eye separation/FOV.

## Packaging Agent

Owns:
- Windows build;
- installer;
- runtime bundling;
- model manager;
- clean-machine testing.

## QA Agent

Owns:
- automated tests;
- visual regression;
- VRAM stress tests;
- release gates.

## Security/License Agent

Owns:
- license matrix;
- attribution;
- dependency audit;
- model redistribution policy;
- secret scanning.

Agents may block a release when a mandatory gate fails.
