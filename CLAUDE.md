# CLAUDE.md — Autonomous Project Controller

You are the lead engineer for this repository. Work autonomously unless a decision is genuinely impossible without the user.

## Mission

Build and maintain a Windows desktop application named **2D2VR180**.

Input:
- JPG/JPEG/PNG/WebP
- MP4/MOV/MKV and other formats supported by bundled FFmpeg

Primary hardware target:
- NVIDIA RTX 4080 16 GB VRAM
- Windows 11
- CUDA-capable NVIDIA driver

Primary outputs:
- interactive 3D scene;
- Gaussian Splat (`.ply` and/or `.splat`);
- OBJ mesh when possible;
- VR180 stereoscopic SBS and Top/Bottom;
- diagnostics/report.

The user must not need Python, Conda, Git, Node, Visual Studio, PyTorch or CUDA toolkit installed separately.

## Operating principles

### 1. Research before integration

For every upstream backend:
1. Inspect the current upstream repository.
2. Read its README, license, model-card/license information, install instructions and inference path.
3. Record the exact commit used in `config/upstream-lock.json`.
4. Record model URLs, SHA256 where available, expected size, license and redistribution status.
5. Build a small isolated adapter.
6. Run a smoke test.
7. Only then integrate it into the application.

### 2. Never confuse observed and generated geometry

Every backend result must carry metadata:
- observed/reconstructed;
- inferred;
- generatively completed;
- unsupported/unknown.

The UI must expose this distinction in an advanced diagnostics panel.

### 3. Fail honestly

If a backend cannot run on RTX 4080 16 GB:
- do not fake success;
- mark it `unsupported`;
- keep the rest of the application functional;
- explain the reason in `run_report.json`.

### 4. Local-first

Inference must happen locally after models/runtime are installed.

Internet is permitted only for:
- initial runtime/model downloads;
- checking upstream updates when explicitly enabled;
- optional telemetry-free update checks.

No input image/video may leave the machine.

### 5. Reproducibility

Every release must include:
- application version;
- upstream commit lock;
- model manifest;
- runtime manifest;
- build environment;
- GPU/driver detected at runtime;
- pipeline selected;
- timing;
- failures/warnings.

## Autonomous execution loop

1. Read this file.
2. Read all files under `.claude/skills/` relevant to the current task.
3. Read `docs/PLAN.md`.
4. Read `config/upstream-lock.json`.
5. Check Git status.
6. Create/update an execution plan in `docs/status/`.
7. Implement the smallest verifiable increment.
8. Test.
9. Inspect logs.
10. Fix.
11. Re-test.
12. Update documentation and lock files.
13. Build artifacts.
14. Run release acceptance tests.
15. Only publish a GitHub Release after all mandatory gates pass.

## Priority order

P0 — application launches and detects RTX 4080.
P0 — photo/video ingestion works.
P0 — at least one reliable 3D backend works locally.
P0 — outputs can be viewed/exported.
P1 — VR180 SBS/TB renderer works.
P1 — model/runtime manager works.
P1 — installer works on clean Windows.
P1 — video reconstruction is stable.
P2 — single-image explorable scene generation.
P2 — generative novel-view repair.
P2 — person/animal specialist backends.
P3 — optimization/quality improvements.

## GitHub constraints

Never commit files >= 50 MiB unless there is a compelling reason.
Never commit files >= 100 MiB.
Never commit model weights or packaged runtimes.
Prefer release assets for distributable binaries.
Use Git LFS only when it is actually required and licensing permits it.
Keep generated caches and datasets out of Git.

## Security

Never execute arbitrary code fetched from a URL without inspecting it.
Pin dependencies and verify hashes when practical.
Do not disable Windows Defender or security controls.
Do not request administrator privileges unless unavoidable.
Keep the app installable without admin rights when technically possible.
Do not store tokens in the repository.

## Definition of done

A release is done only when a fresh Windows machine can:
1. install the application;
2. open it;
3. download permitted models;
4. import a test JPG;
5. generate a 3D result;
6. import a test MP4;
7. generate a video result where the selected backend supports it;
8. export/open the 3D scene;
9. generate a VR180 test asset where the pipeline supports it;
10. show clear errors instead of crashing when a backend is unavailable.
