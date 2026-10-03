# 2D2VR180 1.0.0-rc28 — finer AI views in Quality mode

## What's new in rc28

- **Quality mode draws every AI view in 8 steps instead of 4.** It uses lightx2v's 8-step Lightning LoRA for
  Qwen-Image-Edit-2511 (Apache-2.0, 850 MB).
  - It installs with the Qwen component. If you already have Qwen, install *Qwen-Image-Edit Lightning LoRA
    (8 steps, Quality)* from Settings → Components.
  - Without it, Quality keeps using 4 steps.
  - Generating the views takes about twice as long in Quality. Fast and Auto are unchanged.
- **Camera refinement also covers your own photos.** When you give several real photos (up to 32), training
  now also corrects their cameras slightly. The first photo stays fixed as the reference. Long videos are
  unchanged, because their many overlapping frames already pin the cameras down.

## What's new in rc27

- **The cameras of the AI views are refined while training.** Depth Anything 3 / VGGT estimate where each
  AI view was "taken" from, always with a small error. In a 3D built from only 4 views, that error shows
  up as ghosting and blur.
  - Training now also corrects each AI view's camera slightly, until the view lines up with the 3D. Your
    photo's camera stays fixed as the reference.
  - In the test scene this gave +3 dB sharpness on the photo's view and a lower reprojection error.
  - Results lists how much each camera was corrected. Large corrections (several degrees) point to an AI
    view that is not a clean camera move.

## What's new in rc26

- **Colour drift of the AI views is corrected.** AI-generated views often come out slightly darker,
  brighter or warmer than your photo. Before, the 3D averaged that drift into its colours.
  - Training now learns a small colour correction for each AI view. Your photo is never corrected and
    stays the colour reference.
  - The splat keeps your photo's colours. Results lists the correction found for each view.
- **The OBJ mesh has the same volume as the splat.** The mesh used to be built only from the photo's depth
  map, which gave a relief. It is now fused from the trained splat's depth and colour seen from every view,
  including the AI angles. The old method is kept as a fallback.

## What's new in rc25

- **Turntable preview of every multi-view 3D result.** Right after training, the trained splat is rendered
  from a camera that swings 30° left and right of your photo, with a slight rise and fall. It is saved as
  `export/turntable.mp4` (3 s, 24 fps).
  - This lets you judge the real volume (and spot flat or broken areas) without putting on the headset.
  - Open it with **Results → Play turntable**.
  - It is a preview only: if rendering fails, the result is unaffected and a warning explains why.

## What's new in rc24

- **Results → Rebuild with my picks…** For a "Real 3D from one photo" result:
  - For every AI angle you pick which candidate to use, or untick the angle to leave it out. The automatic
    choice is preselected, with a thumbnail.
  - The 3D is then rebuilt from your photo and those views by the multi-view engine. Nothing is generated
    again, so it takes minutes, not the full run.
  - Your picks are still labelled as AI views: they weigh less than your photo and are marked *generative*.
- The multi-view engine has a dry-run mode, so the whole new flow (candidate selection, Depth Anything 3,
  MoGe-2 priors, confidence maps) can be checked end to end without a GPU.

# 2D2VR180 1.0.0-rc23 — weak AI views are retried automatically

## What's new in rc23

- **Weak AI views are retried automatically ("Real 3D from one photo", Auto and Quality).**
  1. Right after Qwen makes the candidates, the multi-view engine scores each one against your photo.
  2. If even the best candidate of an angle does not behave like a camera move, 2 more candidates are made for
     that angle only, and the best one is chosen again.
  - `job.log` lists each angle's consistency score (lower is better).
- **Fairer scoring.** A different overall brightness or white balance in an AI view no longer counts as a
  change; only real differences in shape, pose or expression do.
- **The user guide** describes the new Create page and the "Real 3D from one photo" flow.
- *Qwen-Image-Edit + Multiple-Angles* is now part of the recommended components, because the default option
  needs it.

# 2D2VR180 1.0.0-rc22 — cleaner splats in VR, AI views overview

## What's new in rc22

- **Cleaner splats when you move your head in VR.** After training, the multi-view engine removes splats that
  only show up as junk from new angles:
  - splats no training view ever saw;
  - nearly transparent ones;
  - oversized blobs;
  - isolated floaters.
  `job.log` says how many were removed.
- **AI views overview.** Every generative job writes `export/generated_views_sheet.jpg`: your photo, then each
  angle with all its candidates, their consistency score (lower = closer to a pure camera move) and which one
  was used (green) or dropped (red). Open it from **Results → AI views overview**. The result details also
  show the engines used and the scores.

# 2D2VR180 1.0.0-rc21 — tuned for "Real 3D from one photo"

## What's new in rc21

- **The photo's full detail.** With photo + 3 AI views the splat is trained at up to 1280 px (Auto) or 1600 px
  (Quality) instead of 1024, so the front view keeps more of the photo's detail.
- **Steadier colours when you move your head.** Colour is learnt with spherical harmonics of degree 1 instead
  of 3. With only four views, higher degrees overfit into colour flicker and tinted patches between the
  views.
- **Inconsistent AI views are dropped.** If even the best candidate of an angle does not behave like a camera
  move of the photo, that view is left out instead of teaching the 3D a different pose or face. At least one
  AI view is always kept, and `job.log` says which view was dropped and why.
- Fix: with Depth Anything 3, views could end up with too few initial points (different confidence scale
  than VGGT).

# 2D2VR180 1.0.0-rc20 — Depth Anything 3 camera engine

## What's new in rc20

- **Depth Anything 3 (Nested Giant-Large 1.1) as the multi-view camera engine.** Most of "Real 3D from one
  photo" depends on where each view was taken and how deep each pixel is, and DA3 estimates both better than
  VGGT: on its authors' benchmark, poses are 35.7% more accurate and geometry 23.6% better. It also gives
  real-world scale directly.
  - It is used automatically when installed: Components → *Depth Anything 3* (~6.8 GB; non-commercial
    licence, like VGGT).
  - It handles up to 32 views of the same shape. Long videos and mixed portrait/landscape photos keep using
    VGGT.
  - It also chooses the most consistent generated candidates.
  - `run_report.json` and `job.log` show which camera engine was used.
- **The Multi-view engine must be installed again** (Depth Anything 3's code was added). The app shows it as
  needing an update.

# 2D2VR180 1.0.0-rc19 — sharper, more solid multi-view training from few views

## What's new in rc19

This release is for "Real 3D from one photo", and for any multi-view job with up to 24 photos.

- **Confidence maps for generated views.**
  - The photo is reprojected in 3D into each AI view and compared pixel by pixel, after removing the overall
    brightness and colour difference.
  - Regions that contradict the photo get almost no weight in training: a slightly turned head, a moved
    hand or a different expression.
  - What only the AI view shows (the sides, the top) keeps full weight.
  - `job.log` says how much of each view was down-weighted.
- **Sharp depth priors.** MoGe-2 depth of every view, scaled onto the multi-view geometry, guides the
  rendered depth during the first 60% of training. With only 4 views this avoids floaters and flat or doubled
  surfaces between views.
- **Denser, sharper start.** The initial point cloud comes from those aligned MoGe-2 depths at full training
  resolution, instead of VGGT's coarser 518-pixel depth.
- When training runs out of GPU memory, the failed attempt's memory is now really freed before retrying.
- Research behind these changes: VidSplat, ReconX and confidence fusion for generated views; FSGS, D²GS and
  HBSplat for depth priors (see `docs/research/single-image-splat.md`).

# 2D2VR180 1.0.0-rc18 — simpler Create page

## What's new in rc18

- **A simpler Create page.** One question, *What do you want to make?*:
  - **Real 3D from one photo (recommended).** AI makes 3 views (45° left, 45° right, from above). The
    multi-view engine then reconstructs real volume and scale from all four, as a trained splat. In Auto and
    Quality the most consistent candidate per angle is chosen automatically.
  - **Quick 3D.** Depth from the photo alone; takes seconds.
  - **360° around the subject (experimental).** The sides, back, top and bottom are invented and
    assembled with sharp fusion.
  - **Custom.** Every setting under *Advanced options*.
- **Quality and VR180 are on one line.** Layouts, projection, backend, generative mode, engine, assembly,
  video mode, combining photos and 4D sequence are all folded into **Advanced options**.
- **A clear warning before you start.** If the chosen option needs a component that is not installed yet (for
  example Qwen-Image-Edit or the multi-view models), the yellow banner says so.
- Radio buttons are easier to see in the dark theme.
- New research and roadmap document: `docs/research/single-image-splat.md`.

# 2D2VR180 1.0.0-rc17 — 3 views (Qwen) → multi-view, most consistent view chosen automatically

## What's new in rc17

- **New mode "3 views (Qwen)"** (replaces Stereo pair). Qwen-Image-Edit with the Multiple-Angles LoRA makes three
  views of your photo: **45° to the left, 45° to the right, and a high-angle shot** from 30° above, looking
  down. The photo and those three views go to the multi-view engine as a **trained splat**.
- **Automatic consistency check.** Qwen redraws the image, so a view can come out with a slightly turned head
  or a different expression. In Auto, 2 candidates are made per angle; in Quality, 3; in Fast, 1. Then:
  1. The multi-view engine (VGGT) places the photo and each candidate in 3D.
  2. It reprojects the photo into the candidate's view and measures how much does not match a pure camera
     move.
  3. It keeps the most consistent candidate.
  - The choice and its scores are written to `job.log` and `run_report.json`.
- **The Multi-view backend can be chosen for a single photo** when a generative mode is on. The views are
  generated first, then reconstructed together with the photo as a trained multi-view splat. With generative
  3D off, it explains that it needs more photos.
- **Stereo pair removed.**

# 2D2VR180 1.0.0-rc16 — stereo pair from one photo (Qwen)

## What's new in rc16

- **New generative mode: Stereo pair (Qwen).** Qwen-Image-Edit makes exactly **two** images of your photo: one
  seen from slightly to the left and one from slightly to the right (about 8° each).
  - The instruction asks to change nothing else: same pose, expression, gaze, clothing, lighting and
    background.
  - The camera-angle LoRA is not used (its smallest step is 45°). The Lightning LoRA is still used.
  - The photo and the two views go to the multi-view engine as a **trained splat**: three almost identical
    views, so nothing gets averaged into blur.
- **Fewer files with Wan + sharp fusion.** Only the key frames used by the fusion are saved now: 7 with 360°
  capture, not every second frame of every shot.

# 2D2VR180 1.0.0-rc15 — Qwen engine fix

## What's new in rc15

- **Fix (Qwen engine):** the views stopped right after the prompts were encoded, with *"When using the offline
  mode, you must specify a weight_name"*. The two LoRAs are now loaded directly from their files; inference
  never touches the network. No reinstall needed (the rc14 Generative engine is still valid).

# 2D2VR180 1.0.0-rc14 — 360° photo capture, Qwen-Image-Edit engine

## What's new in rc14

- **360° photo capture (new generative mode).** A few widely spaced views starting from your photo: 45°, 90°,
  135°, 180° (the back is invented), 270°, overhead and from below. That is 7 views in Fast and Auto. Quality
  adds 225°, 315° and four raised diagonals (13 views). They are assembled with sharp fusion.
- **New generative engine: Qwen-Image-Edit-2511 + Multiple-Angles LoRA** (Alibaba Qwen and fal, Apache-2.0).
  It redraws your photo from each requested camera angle as a sharp ~1 megapixel image: sharper than video
  frames, and only 4 sampling steps per view thanks to the Lightning LoRA.
  - **Memory:**
    - The 20B transformer (GGUF, 5-bit) streams block by block from RAM to the GPU.
    - The text encoder (Qwen2.5-VL-7B, 4-bit) runs first, in its own process, then releases everything.
  - **To install:** Components → *Qwen-Image-Edit-2511 + Multiple-Angles*, about 33 GB.
  - **To use it:** pick *Generative engine*: Automatic (Qwen first), Qwen, Wan 2.2 or Stable Virtual Camera.
    Qwen does 360° capture, Around the subject and Wide orbit. Explore and Spiral use Wan 2.2.
- **Wan 2.2 also does 360° capture.** It films an orbit to 180°, a turn to 270°, a crane shot above and one
  below, and keeps exactly the frame at each requested angle.
- **The generative engine must be installed again** (diffusers 0.37.1, peft, gguf and bitsandbytes were
  added). The app shows it as needing an update.

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
| `2D2VR180-1.0.0-rc28-setup.exe` | Installer — per user, no administrator rights |
| `2D2VR180-1.0.0-rc24-portable-win64.zip` | Portable version — unzip and run `2D2VR180.exe` |
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
