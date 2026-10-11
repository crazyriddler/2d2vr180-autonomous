# Video → 4D Gaussian Splatting (research and plan, October 2026)

Owner request (October 2026): videos are useless today, because only one frame really becomes 3D. Videos
should become **4D Gaussian splats** (3D that moves over time), with a **player** for them. The current VR180
video path is to be removed and rebuilt on top of 4DGS. This comes after the single-photo work (rc41–rc46).

Target: RTX 4080 16 GB, 32 GB RAM, Windows 11, everything local.

## What exists today

| Path | What it does | Limitation |
|---|---|---|
| Moving camera, static scene | Multi-view engine on up to 80 keyframes → one static splat | Anything that moves is averaged away or ghosted |
| Fixed camera, moving people | Per-frame MoGe-2 RGB-D splats, depth stabilised over time → VR180 video; optional `.ply` per frame | 2.5D per frame (front surface only), no shared background, no player |
| Fixed camera, nothing moves | Sharpest frame only | — |

## Building blocks found (2025–2026)

**Cameras and depth for videos with moving content**
- **MegaSaM** (Google/Cornell, Apache-2.0): camera poses, focal length and consistent video depth for casual
  dynamic videos. It is the standard input for monocular 4D methods. Python/PyTorch with CUDA extensions
  (DROID-SLAM-based).
- Reported VRAM: about 10 GB for 30 frames, 24 GB for 100 (third-party report, to be measured). It needs
  windowing on 16 GB.

**4D reconstruction from one video**

| Method | Type | Licence | Notes |
|---|---|---|---|
| Shape of Motion (2024) | per-scene optimisation: SE(3) motion bases | research code | Needs MegaSaM/depth + 2D tracks + masks; 20–60 min per clip |
| MoSca (2024) | per-scene optimisation: motion scaffold | research code | Similar inputs; strong on casual videos |
| 4DGT (Meta, NeurIPS 2025) | feed-forward 4D Gaussian transformer, 64-frame rolling windows | weights CC BY-NC-SA 4.0 (14.5 GB) | Needs posed frames (MegaSaM); trained on real monocular videos; seconds per window |
| NeoVerse (2026), 4Real-Video-V2, Lyra 2 | generative 4D world models | various | 40–80 GB class, not for 16 GB |

**Players and formats**
- No standard 4D splat format yet (MPEG has an exploration). Two practical options:
  1. **Per-frame splat sequence:** one `.ply`/`.splat` per frame, played like video. Universal and simple,
     but large.
  2. **Motion-parametrised splats:** for example splaTV's `.splatv` (MIT, browser viewer with WebXR,
     Spacetime-Gaussians-style polynomial motion and temporal opacity). Compact and streams after one
     download.
- Our own viewers (GL viewer in the app; WebXR viewer served to the browser) can play option 1 directly.
  Option 2 needs a shader that evaluates motion per Gaussian.

## Plan and status

| Step | Content | Status |
|---|---|---|
| 1 | **4D player:** the app's 3D viewer and the WebXR (VR) viewer play a 4D scene (play/pause, timeline, loop) | **Done in rc49.** `.splatv` (splaTV format) with our own WebGL2/WebXR player derived from splaTV (MIT); the app's viewer plays the same scene with a timeline |
| 2 | **Static background + moving foreground** in one world frame | **Done in rc49**, with VGGT instead of MegaSaM (see below) |
| 3 | **Moving Gaussians for the moving parts** | **rc50: Spacetime-Gaussian optimisation** (workers/motion_trainer.py, below). 4DGT was ruled out for now: flash-attn + apex (no practical Windows build), >= 16 GB *free* VRAM, inputs prepared from Aria recordings, non-commercial weights (14.5 GB). Shape of Motion needs 2D tracks (TAPIR / CoTracker), SAM masks and 20-60 min per clip: the next candidate for real volume (motion bases + tracks), after measuring rc50 on real footage |
| 4 | **New VR180 video** from the 4D scene; old per-frame VR180 path removed | **Done in rc49** |

### What rc49 builds (workers/video4d_worker.py)

1. **Frames:** every frame at a fixed rate (by quality: 8 / 12 / 15 per second, 8 / 12 / 15 s; adjustable).
2. **Cameras:** VGGT on all frames, chunked and Sim(3)-stitched (the multi-view engine's code). A camera that
   neither moves (1 % of the scene depth) nor turns (1°) is held exactly still. MegaSaM was the research choice
   for dynamic videos, but it needs DROID-SLAM CUDA extensions with no Windows build and ~24 GB for 100 frames;
   VGGT is installed already and robust when most of the image is still. To be revisited if poses drift on
   videos dominated by movement.
3. **Depth:** MoGe-2 per frame with the camera's field of view, aligned (scale + shift, robust) onto VGGT's
   joint geometry; metric scale from MoGe-2.
4. **What moves:** each frame is reprojected into frames 0.5, 1.5 and 3 s away and into five frames spread
   over the clip. A pixel moves if any of them sees free space where its 3D point is (it is not permanent),
   or if most of them see another colour at the same depth (best match within one pixel; global exposure
   removed). No evidence (hidden, out of view, depth edge) counts as still. Masks are cleaned (opening,
   closing, small blobs removed, holes filled) and grown by about 1 %. Depth is then aligned again on still
   pixels only; with a fixed camera all frames share one median background depth.
5. **Background:** one 3D Gaussian splat trained on all frames with the moving pixels weighted 0 and the
   still pixels' depth as prior: what a passer-by hides in one frame is learnt from the others.
6. **Moving parts:** per frame, splats on a pixel grid of the moving region (budget 25k / 40k / 60k per frame),
   flat discs facing that frame's camera, at that frame's aligned depth, visible only at that frame's time.
   2.5D: the side the camera saw. Labelled INFERRED.
7. **Outputs:** `scene.ply` (background, for any viewer), `scene.splatv` (whole 4D scene, capped at 4 M splats
   for the browser), full-precision `dynamic.npz` + `scene4d.json`, a TSDF mesh of the background, a preview of
   the moving masks.
8. **VR180 video:** each frame rendered (gsplat, background + that frame's moving splats) from the levelled
   camera of that frame; a moving camera's path is smoothed (Gaussian, σ = fps / 4 frames).

### rc50: smooth motion (workers/motion_trainer.py)

- Seeds: the moving splats of every second frame (and the last). Each Gaussian lives about one keyframe
  interval either side of its keyframe (temporal opacity), with a velocity from where its group went:
  groups are separated by DBSCAN, moved centre to centre, refined by translation-only ICP, then matched
  point by point (matches farther than 3x the point spacing are ignored). Monocular photometric optimisation
  alone does not find the motion from zero velocity (verified on a synthetic test), so this seed is what
  makes it work; Shape of Motion uses 2D tracks for the same reason.
- Optimised on every frame, background fixed and drawn together: L1 + SSIM on the moving pixels (others
  weighted 0.2), relative depth L1 on the moving pixels, local rigidity (8 neighbours from the seed keep
  their distances at any time), small accelerations. Parameters: position, velocity, acceleration,
  rotation + angular velocity, scale, opacity, colour, temporal scale. 2000 / 4000 / 6000 steps by mode.
- Output in the .splatv motion layout (velocity, acceleration, cubic term 0); the scene plays continuously
  and the VR180 video is rendered at 24 fps. Falls back to per-frame splats on any failure.

Not yet measured on the RTX 4080: speed, VRAM, mask quality on real footage (shadows, reflections, lighting
changes), and pose stability on videos where most of the image moves.

Each further step: research the upstream, record commit, licence and model hashes, build an isolated adapter,
smoke test, then integrate (CLAUDE.md).
