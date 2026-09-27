# Initial Claude Code Prompt

You are taking ownership of the repository **2D2VR180**.

Your job is to turn this repository into a working, distributable Windows application for an NVIDIA RTX 4080 16 GB.

Do not merely write a plan. Execute the plan autonomously.

## Read first

Read:
- `CLAUDE.md`
- `AGENTS.md`
- `docs/PLAN.md`
- `docs/ARCHITECTURE.md`
- `docs/REPOSITORY_MATRIX.md`
- `docs/GITHUB_STORAGE.md`
- `docs/QUALITY_GATES.md`
- every `.claude/skills/*/SKILL.md`
- every `.claude/agents/*.md`

## Mission

Build an application where the user can:

1. drag a JPG/PNG/WebP photo or MP4/MOV video into the app;
2. choose Auto / Quality / Fast;
3. click Generate;
4. obtain an explorable 3D reconstruction;
5. export Gaussian Splat and OBJ when possible;
6. generate stereoscopic VR180 SBS/TB when the reconstruction supports it;
7. do all inference locally on the RTX 4080;
8. avoid installing Python/CUDA/Conda/Git/Node/Visual Studio/PyTorch manually.

## Autonomous policy

You are allowed to:
- clone upstream repositories;
- inspect their code;
- create isolated environments;
- compile native/CUDA components;
- download permitted model weights;
- write adapters;
- write tests;
- build Windows installers;
- run benchmarks;
- revise the architecture when evidence demands it.

You must:
- keep upstream code isolated;
- pin exact commits;
- maintain license records;
- never commit large model files;
- never publish restricted weights;
- test on a CUDA GPU in the cloud before claiming local support.

## First task: reconnaissance

Before writing substantial application code:

1. verify every repository URL in `config/upstream-lock.json`;
2. resolve the latest usable commit;
3. inspect license and model license;
4. inspect installation;
5. determine whether Windows is feasible;
6. determine VRAM requirements;
7. run minimal inference where feasible;
8. record results.

Add newly discovered superior repositories to the matrix rather than blindly keeping the initial list.

## Initial implementation order

### Milestone 1
Hardware diagnostics + project skeleton + job system.

### Milestone 2
Working photo backend.

### Milestone 3
Working video backend.

### Milestone 4
3D viewer/export.

### Milestone 5
VR180 renderer.

### Milestone 6
Model manager.

### Milestone 7
Installer.

### Milestone 8
Generative novel-view repair.

### Milestone 9
Specialist human/animal pipelines if justified.

## Important technical decisions

Prefer existing working systems over reimplementing research papers.

Use recon3d as a reference for a practical video pipeline because it already connects VGGT, metric depth and gsplat and supports video/images, mesh and several export formats.

Use Splatline as an architectural reference for a pluggable multi-backend desktop workflow.

Use VGGT as a geometry backend where appropriate.

Evaluate One2Scene for single-image explorable scenes.

Evaluate SHARP only with explicit license handling.

Evaluate LongSplat and DepthSplat for video quality.

Evaluate GSFixer/GSFix3D for novel-view repair.

Do not assume any candidate is production-ready until tested.

## Photo vs fixed-camera video

A fixed-camera video does not automatically provide more geometric information than a photo.

If the scene is static, treat redundant frames efficiently.

If people/animals move, consider whether their changing poses/views make a dynamic reconstruction pipeline more appropriate.

The pipeline selector must not waste 10 minutes processing 500 nearly identical frames.

## Output truthfulness

Every job must generate:

`run_report.json`

containing:
- input;
- backend;
- exact commits;
- model revisions;
- hardware;
- VRAM peak if measurable;
- runtime;
- output;
- observed/inferred/generated coverage;
- warnings.

## Packaging

Do not put models into Git.

Do not put binaries over GitHub's normal file limits into the repository.

Use GitHub Releases for application binaries. If a release asset is too large, split the distribution logically or host/download the model/runtime externally.

The repository should remain source-first and cloneable.

## Release requirement

At the end, create:

- Windows installer;
- portable ZIP;
- SHA256 checksums;
- release manifest;
- release notes;
- reproducible build instructions.

Before publishing:
- run clean Windows installation test;
- run photo test;
- run video test;
- run VR180 test;
- run error-path tests;
- run license/security audit.

If a component cannot be shipped, remove it from the default release and expose it as an optional research backend.

## Definition of success

The user should be able to download one release, install it, point it at a photo/video, and get a useful 3D/VR result without becoming a Python/CUDA/ML engineer.

Do not stop at documentation. Build the product.
