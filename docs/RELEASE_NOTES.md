# 2D2VR180 1.0.0-rc1 — release candidate for RTX 4080 testing

Turn photos and videos into explorable 3D scenes (Gaussian splats, meshes) and VR180 stereo,
entirely on your own Windows PC. No Python, CUDA toolkit, Git or other developer tools needed.

> **Release candidate.** Everything below was built and tested automatically on Windows
> (GitHub-hosted runners: real runtime installation, real model downloads, real inference on the
> **CPU**). The runners have no NVIDIA GPU, so **GPU inference and the GPU splat renderer have not
> yet run on an RTX 4080**. Please report anything that fails, with the text from
> *Diagnostics → Copy to clipboard* and the job's `run_report.json`.

## Download

| File | What it is |
|---|---|
| `2D2VR180-1.0.0-rc1-setup.exe` | Installer — per user, no administrator rights |
| `2D2VR180-1.0.0-rc1-portable-win64.zip` | Portable version — unzip and run `2D2VR180.exe` |
| `checksums.sha256` | SHA256 of every file |
| `release-manifest.json` | Exact source commit, upstream commits, model/runtime manifests, tool hashes |

Start the app → the welcome dialog offers **Install recommended components** (engine ~5.5 GB +
models ~1.6 GB, downloaded once from the original publishers). See `docs/USER_GUIDE.md`.

## Features

- **Create**: drag & drop photos or videos, Auto / Quality / Fast, queue several jobs, cancel any time.
- **Photo → 3D**: MoGe-2 (metric depth → surface-aligned Gaussian splats + textured OBJ mesh),
  Depth-Anything-V2 Small (Apache-2.0 fallback), Apple SHARP (optional; research licence).
- **Video**: automatic analysis — fixed camera & static scene → sharpest frame only; fixed camera with
  moving people/animals → per-frame 3D and a VR180 video of the motion; moving camera → multi-view
  reconstruction with recon3d (VGGT poses + MoGe-2 metric scale + gsplat training + TSDF mesh), rendered
  along the recovered camera path.
- **Exports**: `.ply` (standard 3DGS), `.splat`, `.obj`+`.mtl`+texture.
- **VR180**: rendered from the 3D scene (never by shifting pixels); side-by-side and top/bottom; JPEG stills
  and H.264 MP4 with Spherical Video V2 (`st3d`/`sv3d`) metadata; coverage masks; configurable eye separation
  and resolution; GPU splat renderer (gsplat) with automatic CPU fallback; optional background hole filling.
- **Honest output**: every splat is labelled observed / inferred; VR180 reports the real content field of view
  and how much is unknown; every job writes `run_report.json` (input hash, backend, upstream commits,
  model revisions, hardware, VRAM peak, timings, coverage, warnings) — also for failed/cancelled jobs.
- **Components** page: install/remove/verify engines and models with licence display, progress, resume,
  SHA256 verification. **Settings**: licence profile (personal/research vs commercial), renderer, hole filling,
  eye separation, resolutions, CPU mode, output folder, HF token. **Results** page with previews and one-click
  open/play/export. **3D viewer** with orbit/pan/zoom and provenance colouring. **Diagnostics** page.

## Verified automatically (no GPU)

- 70+ unit/integration tests on Windows and Linux (scene I/O, VR180 geometry and eye order, video analysis,
  model manager incl. resume/corruption, job pipeline incl. cancel/OOM/missing model/no GPU, GUI).
- Packaged app: build, launch, acceptance checks, silent install → run without Python → uninstall.
- Photo engine installed by the packaged app on Windows (torch 2.8.0+cu128, MoGe, SHARP, transformers).
- Video engine installed by the packaged app on Windows (torch 2.4.1+cu124, prebuilt gsplat 1.5.3, VGGT, recon3d).
- Models downloaded through the model manager and verified against pinned SHA256.
- **Real inference on Windows (CPU)** through the packaged app:

  | Run | Backend | Splats | Source-view PSNR |
  |---|---|---|---|
  | Photo | Depth-Anything-V2 | 251k | 24.0 dB |
  | Photo | MoGe-2 (fast) | 254k | 23.6 dB |
  | Photo | Apple SHARP | 1.18M | 27.9 dB |
  | Photo, commercial profile | auto → Depth-Anything-V2 | 251k | 24.0 dB |
  | Fixed-camera video → VR180 video | MoGe-2 | 98k/frame | 22.1 dB |

  Bugs found and fixed by these runs: download progress crash, Windows console encoding crash,
  runtime dependency conflicts (opencv/numpy), missing gtsam wheels on Windows.

## Known limitations

- **Not yet run on an NVIDIA GPU** (see above).
- A single photo shows only what the camera saw: VR180 from a photo is a ~60–80° window inside the 180° frame;
  the rest is black and reported as unknown. No generative completion is shipped (GSFixer / GSFix3D /
  One2Scene are unlicensed, non-commercial or not installable on Windows without a compiler).
- Moving-camera video needs 12 GB+ VRAM; on Windows no GTSAM wheel exists, so long videos use chunk stitching
  without factor-graph refinement.
- Per-frame (fixed-camera) VR180 video reconstructs frames independently; depth can flicker.
- The GPU renderer uses the video engine's prebuilt gsplat; without the video engine VR180 uses the CPU renderer.

## Licences

App: MIT. Bundled FFmpeg: GPL-3.0-or-later (separate executable; source link in THIRD_PARTY_NOTICES.md).
Models are downloaded from their publishers after you accept their licences: MoGe-2 (MIT),
Depth-Anything-V2 Small (Apache-2.0), SHARP (Apple research licence, non-commercial), VGGT-1B (non-commercial).
