# 2D2VR180 1.0.0-rc13 — Wan 2.2 memory: one process per attempt, fewer frames

## What's new in rc13

- **Fix: Wan 2.2 still ran out of GPU memory and filled the 32 GB of RAM (rc12, Quality).** Each failed
  memory setting left about 8 GB of VRAM and a lot of RAM behind inside the same process. Now:
  - **Separate process per attempt.** Each memory setting runs in its own process, and only exiting the
    process frees everything for certain.
  - **Nothing is redone.** Shots that are already finished, and the encoded prompt, are reused.
  - **16 GB cards start directly with FP8 weights.** That is the setting that fits; a 24 GB+ card still
    starts with full precision.
- **Fewer frames for sharp fusion.** Fusion only uses 4 key views per shot, so the shots are now 25 frames
  (Fast), 33 (Auto) or 49 (Quality), instead of up to 81. Faster and lighter.

# 2D2VR180 1.0.0-rc12 — generative 3D: sharp fusion, views from above and below

## What's new in rc12

- **Sharp fusion — new default for generative 3D.** rc11 trained one splat on every generated frame. The
  frames never agree perfectly, so training averaged them: soft from the side, broken from above or below.
  Now:
  1. Wan 2.2 films fewer, shorter shots.
  2. VGGT places four key views per shot in 3D.
  3. MoGe-2 gives each key view the same crisp per-pixel geometry as the single-photo mode.
  4. The photo is kept exactly as it is. The other views add only surfaces it does not show (sides, top,
     underside, the background behind the subject). Where two views disagree, the earlier one wins instead of
     being averaged.
  - No training step, so it is sharp from every angle that a view covers, and faster.
  - The previous behaviour is still available as *Generative 3D assembly → Trained splat*.
- **Views from above and below.** *Around the subject* now films four shots: 45° to the right, 45° to the
  left, 35° from above, and 25° from below. *Wide orbit* does the same at ±100°, 50° above and 30° below.
- `run_report.json` now includes the backend details: assembly, engine, views, splats.
- CI: the real VGGT + MoGe-2 + fusion pipeline runs on Windows (CPU) on every integration build.

# 2D2VR180 1.0.0-rc11 — Wan 2.2 fits in 16 GB VRAM / 32 GB RAM

## What's new in rc11

- **Fix: Wan 2.2 ran out of GPU memory and filled the RAM on an RTX 4080 with 32 GB.**
  - The 11 GB text encoder (umT5-XXL) now runs once, at the start, and is then released. Before, it stayed in
    VRAM while the photo was encoded and in RAM for the whole run.
  - The video VAE leaves the GPU while sampling, and the camera-control tensor is half the size.
  - When a setting does not fit, the failed attempt's memory is now really freed before the next one. Before,
    every retry stacked another copy of the model on top, until the RAM was full.
  - No reinstall needed; just update the application.
- CI: new Windows check that assembles the Wan 2.2 pipeline exactly as the worker uses it.


## What's new in rc10

- **New default generative engine: Wan 2.2 Fun 5B Control-Camera** (Alibaba PAI, Apache-2.0). RTX 4080 tests of
  rc9 showed that Stable Virtual Camera deforms people from the very first generated view (extra limbs,
  distorted faces) - a limitation its authors document. Wan 2.2 is a video model trained on real footage:
  it "films" camera moves that start at your photo (one shot to each side for *Around the subject*, wider
  ones for *Orbit*, four for *Explore*), and the multi-view engine reconstructs the frames into one splat.
- **No Hugging Face account needed** any more: Wan 2.2 is a public download (~25 GB, needs ~32 GB of RAM).
  Stable Virtual Camera becomes optional (used only if Wan 2.2 is not installed).
- The generative engine runtime changes: open **Components** and install **Generative engine** again, then
  **Wan 2.2 Fun 5B Control-Camera** (or press *Install recommended*).
- Progress shows each shot and diffusion step with seconds per step and GPU memory; out-of-memory falls back
  automatically to fp8 weights, a lower resolution, or sequential offloading.

## rc9

## What's new in rc9

- **Generated views of people were deformed (RTX 4080 report on rc8).** Changes:
  - generation starts at Stable Virtual Camera's native 576-pixel resolution (rc7/rc8 used 448 to save
    memory); the memory cap still falls back to 512/448/384 automatically if needed;
  - always 50 diffusion steps (Fast used 30); Fast now only generates fewer views;
  - new default path **Around the subject ±60°** (Stable Virtual Camera's figure-of-eight preset): far more
    reliable than a full 360° orbit from one photo, and the best choice for people. *Orbit 360°* remains
    available for objects, but the back side of a person seen from the front is pure invention and often
    deformed.

## rc8

## What's new in rc8

- **Fix (RTX 4080 report on rc7): generative 3D failed immediately** with "Input type (torch.FloatTensor)
  and weight type (torch.cuda.FloatTensor) should be the same". Stable Virtual Camera moves each component
  to the GPU before use; the CLIP encoder now stays on the CPU as intended.
- Stable Virtual Camera's own low-VRAM mode is enabled: the diffusion model and the image decoder take turns
  on the GPU instead of occupying it together.

## rc7

## What's new in rc7

- **Fix (RTX 4080 report on rc6): generative 3D used 18.3 GB on a 16 GB card**, Windows spilled into shared
  system memory and each diffusion step took 53 s even in Fast mode. Now:
  - the GPU memory of the generation and training processes is capped below the card's size, so instead of
    a silent, very slow spill the app gets an out-of-memory signal and retries automatically at a lower
    resolution (generation) or with smaller images and fewer splats (training);
  - the CLIP image encoder of Stable Virtual Camera runs on the CPU (it only sees a few input images):
    ~2.5 GB less GPU memory;
  - the generation resolution follows a pixel budget (576×576 pixels): a 4:3 photo is generated at 597×448,
    16:9 at 683×384, instead of 768×576 / 1024×576.

## rc6

## What's new in rc6

- **Fix: generative 3D looked frozen at "Sampling 0/50".** Stable Virtual Camera's progress bar redraws one
  console line, which the app could not show. Every diffusion step is now reported: pass (1/2, 2/2), chunk,
  step, seconds per step and GPU memory in use.
- **Warning when the GPU memory is full** and steps become slow (Windows then uses shared system memory,
  which is many times slower): close other GPU programs or use *Fast* mode (48 views, 30 steps).
- Removed a harmless PyTorch warning about `expandable_segments` on Windows.
- Windows CI with real models (CPU): VGGT-1B + MoGe-2 camera poses, LaMa inpainting and Stable Virtual Camera
  loading all pass.

## rc5

## What's new in rc5

- **Fix: VR start position.** In the VR viewer you now start at eye level exactly where the camera stood,
  with the horizon levelled (it used to start high above the scene and tilted). The gravity direction is
  estimated from the scene's floors, ceilings and tables; the app's 3D viewer and the VR180 renders use the
  same level orientation.
- **New: several photos → one 3D scene.** Drop photos of the same place taken from different positions:
  the new multi-view engine finds the cameras (VGGT), gives real-world scale (MoGe-2) and trains a real
  3D Gaussian splat from all of them, plus a mesh.
- **New: whole video → one 3D scene.** Videos with a moving camera now use up to 80 frames of the whole
  video (not one frame). A *Video* option lets you force *whole video*, *frame by frame* or *sharpest frame*.
  The previous recon3d pipeline was replaced: it trained full-resolution images with camera parameters
  computed for 518-pixel images.
- **New: Generative 3D from one photo.** Stable Virtual Camera (Stability AI) generates new views of the
  photo along a camera path — *Orbit* (other sides of the subject, 360°), *Explore* (the surroundings, fills
  a VR180) or *Spiral* — and the multi-view engine trains one consistent 3D splat from them. Invented parts are
  labelled *generative* (pink in *Colour by provenance*).
- **New: AI hole filling (LaMa)** for the gaps that appear behind objects in VR180 stereo.
- **Fixed-camera video with moving people/animals:** depth is stabilised over time (less flicker), VR180
  is levelled, and you can save one splat per frame (4D sequence).
- **Components:** every engine and model is part of *Install recommended* (~40 GB). Stable Virtual Camera is
  gated: accept its licence on Hugging Face and paste a token in Settings (the app explains how).
- CLI: `run` accepts several photos or a folder, `--generative orbit|explore|spiral`,
  `--video-mode`, `--export-sequence`, `--no-ai-fill`.

### Verified automatically (no GPU)
- 98 tests incl. the 3DGS training loop on the CPU (reference renderer), multi-view geometry (VGGT input
  mapping, Sim(3) stitching), gravity estimation, generative and multi-photo job pipelines, LaMa
  integration, and the WebXR viewer in Chrome (eye-level start, levelled scene, mouse controls).
- Windows CI integration: real VGGT-1B + MoGe-2 camera poses on the CPU, real LaMa inpainting on the CPU,
  Stable Virtual Camera import with the Windows-safe attention patch.
- **Not yet run on an RTX 4080:** gsplat training, Stable Virtual Camera generation, GPU speeds and memory.

### Requirements for the new features
NVIDIA GPU with 10 GB+ (multi-view) or 12 GB+ (generative 3D); RTX 4080 16 GB is the target.
Multi-view and generative 3D are **non-commercial** (VGGT-1B, Stable Virtual Camera licences); they are
disabled in the *Commercial* licence profile.

## rc4

## What's new in rc4

- **Fix: the camera could not be moved in the VR viewer page.** The splat library switches its mouse controls
  off when WebXR is enabled; the page now attaches its own controls: drag = orbit, right-drag = pan,
  wheel = zoom, arrow keys = pan, R = reset.
- **New: move with the VR controllers.** Left stick = move (in the direction you look), right stick
  left/right = snap-turn 30° (around your head), right stick up/down = rise/sink, A or X = back to the start.
  Implemented by offsetting the WebXR reference space (the scene is never modified).
- Browser tests of the viewer (page loads, mouse moves the camera, locomotion maths) now run in CI.

## rc3

## What's new in rc3

- **Fix: "View in VR" failed when Microsoft Edge is not installed.** The app now finds Google Chrome, Edge or
  Brave (install folders and Windows *App Paths* registry) and opens the viewer there; if none is found it uses
  the default browser and shows the address with a *Copy address* button (VR needs Chrome or Edge).

## rc2

## What's new in rc2 (from RTX 4080 feedback on rc1)

- **Fast 3D viewer**: splats are now drawn on the GPU (OpenGL point sprites, depth-tested). Measured with
  1M splats: the rc1 CPU viewer took ~1.7 s per frame; the GPU view is interactive (a few ms per frame on an
  RTX GPU). A splat-size slider was added. Without OpenGL the viewer falls back to the CPU preview, which now
  renders a lighter preview while dragging (~9× faster) and full quality on release.
- **View in VR (6DoF)**: new button on Results and the 3D viewer. Opens the splat in a bundled WebXR viewer
  (three.js + GaussianSplats3D, MIT; served only to 127.0.0.1) in Chrome or Edge — press ENTER VR with a
  Quest Link / Air Link / Virtual Desktop / SteamVR headset. CLI: `2d2vr180-cli view-vr scene.ply`.
- **Clearer output folder**: a `README.txt` explains every file; app metadata moved into `_2d2vr180\`.
  The VR180 videos for standalone headsets are in `export\vr180\`.

---

# 2D2VR180 1.0.0-rc1

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
| `2D2VR180-1.0.0-rc13-setup.exe` | Installer — per user, no administrator rights |
| `2D2VR180-1.0.0-rc13-portable-win64.zip` | Portable version — unzip and run `2D2VR180.exe` |
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
