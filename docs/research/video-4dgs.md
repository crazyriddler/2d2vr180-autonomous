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

## Plan

| Step | Content | Result for the owner |
|---|---|---|
| 1 | **4D player:** the app's 3D viewer and the WebXR (VR) viewer play a splat sequence (play/pause, timeline, loop) | Today's per-frame sequences can be watched in 3D/VR, moving |
| 2 | **Static background + moving foreground:** MegaSaM cameras and depth for every frame; people/animals masked; the background trained once from all frames (multi-view engine, sharp and complete); the foreground as per-frame splats in the same world frame | A real 4D scene: a solid 3D background, people moving in it |
| 3 | **True 4DGS for the moving parts:** evaluate 4DGT (feed-forward, fast) and Shape of Motion (optimised, best quality) on the RTX 4080; export `.splatv` and/or the sequence | Moving subjects with volume, not only their visible side |
| 4 | **New VR180 video:** stereo frames rendered from the 4D scene along the original camera path (or a stabilised one); the old per-frame VR180 path removed | VR180 videos with real depth and parallax |

Each step: research the upstream, record commit, licence and model hashes, build an isolated adapter, smoke
test, then integrate (CLAUDE.md).
