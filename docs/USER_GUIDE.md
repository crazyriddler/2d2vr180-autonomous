# 2D2VR180 — User guide

2D2VR180 turns photos and videos into explorable 3D scenes (Gaussian splats and meshes)
and stereoscopic VR180 images and videos. Everything runs on your own computer: nothing you
process is uploaded anywhere.

## Requirements

- Windows 10 (build 19041+) or Windows 11, 64-bit
- An NVIDIA GPU with a recent driver (target: RTX 4080 16 GB; 8 GB+ works for photos;
  10 GB+ for videos in 4D, 12 GB+ recommended). Driver 570 or newer is recommended.
- About 8 GB of free disk space for the engine and models (+ space for your results)
- Internet access **only** for the one-time download of the engine and models

You do **not** need Python, CUDA, Git, Conda, Visual Studio or anything else.

## Install

1. Download `2D2VR180-<version>-setup.exe` from the GitHub release (or the portable ZIP).
2. Run it. It installs for your user only; no administrator rights are needed.
3. Start **2D2VR180** from the Start menu.

The portable ZIP works the same way: unzip anywhere and run `2D2VR180.exe`.

## First start: install the components

On first start the app offers to install the recommended components (about 65 GB in total, most of it the
Qwen-Image-Edit model that Real 3D needs). You can also open **Components** at any time and install only what you
need.

| Component | What it enables | Size |
|---|---|---|
| Photo engine | every single-photo reconstruction, fixed-camera video, GPU VR180 renderer | ~5.5 GB |
| MoGe-2 Large / Small | metric depth for photos and video; real-world scale for the multi-view engine | 1.3 / 0.14 GB |
| MoGe-3 Large | Real 3D: depth, field of view and real-world scale of your photo (MoGe-2 is used without it) | 1.5 GB |
| Apple SHARP | best direct single-photo 3D for Quick 3D (**research licence**) | ~2.5 GB |
| Multi-view engine + VGGT-1B | Real 3D, several photos of one place, videos where the camera moves (**non-commercial**) | ~7 GB + 5 GB |
| Generative engine + LaMa | the runtime for Qwen-Image-Edit and Wan 2.2; AI filling of the gaps behind objects in VR180 | ~6 GB + 0.2 GB |
| Qwen-Image-Edit-2511 + Multiple-Angles + Lightning | **Real 3D**: redraws your photo from other camera angles, sharp ~1 MP stills (Apache-2.0) | ~33 GB (needs ~32 GB RAM) |
| Wan 2.2 Fun 5B Control-Camera *(optional)* | exploring the surroundings for VR180; Real 3D when Qwen is not installed (Apache-2.0) | ~25 GB (needs ~32 GB RAM) |
| Depth-Anything-V2 Small *(optional)* | depth model for the Commercial licence profile (Apache-2.0) | 0.1 GB |

Every model shows its licence before downloading. Downloads resume if interrupted and are
verified (SHA256) before use.

## Make something

1. Open **Create** and drop one or more photos or videos (or click to browse).
2. Choose **Auto** (recommended), **Quality** or **Fast**.
3. Choose outputs: VR180 side-by-side and/or top/bottom, VR180 or flat 3D projection.
4. **What do you want to make?** — *Real 3D from one photo* (recommended), *Quick 3D* (seconds, depth from the
   photo alone), *360° capture* (more angles at once) or *Custom*. **Advanced options** show every setting:
   - **VR180 output** — layouts (side-by-side, top/bottom) and projection.
   - **One photo**
     - **AI views** — *Real 3D* (the default) draws your subject from nearby angles, straight from your photo:
       45° left, 45° right and from above (Auto and Quality add one from below). In Auto/Quality 2-3 candidates
       per angle are made and the one that is most exactly your photo seen from there is kept. It is made for
       looking around from the front (VR): no back is invented.
       *360° capture* draws 45°, 90°, 135°, 180°, 270°, overhead and from below at once (Quality adds 225°,
       315° and raised diagonals). *Explore the surroundings* (Wan 2.2) films the view around the photo's
       position so a VR180 is filled. *Off* uses only what the photo shows. In every case the photo and the views
       are reconstructed together into one trained 3D splat.
     - **AI model for the views** — *Qwen-Image-Edit* (recommended) draws every angle as a sharp ~1 megapixel
       image; *Wan 2.2* films the camera moving (smoother, 704 px); *Automatic* uses Qwen when installed.
     - **Reconstruction engine** — *Automatic* picks the best one for the input.
   - **Several photos and videos**
     - **Video length (4D)** — how much of the video becomes 4D: by quality (8 s Fast, 12 s Auto, 15 s
       Quality, recommended) or 5–30 s. Longer takes longer and needs more memory.
     - **Video frames (4D)** — frames per second rebuilt in 3D: by quality (8 / 12 / 15, recommended) or
       6–24. More is smoother motion, but slower and a larger 4D file.
     - **Combine several photos into one 3D scene** — photos of the same place taken from different
       positions are reconstructed together (the more angles, the more complete).
5. Click **Generate**. Jobs run one after another; you can cancel any of them.

When a job finishes, **Results** shows it. From there you can:
- **Explore in 3D** — GPU viewer: orbit (drag), pan (right-drag), zoom (wheel), splat size, colour by provenance.
  A 4D result (video) plays: play/pause and a timeline under the view.
- **View in VR (6DoF)** — look around the 3D splat in a headset connected to this PC (see below). A 4D result
  opens in the 4D player: the scene moves while you walk around it (trigger = play/pause).
- **Open VR180 image / Play VR180 video**.
- **Open folder** — the `export` folder contains:
  - `scene.ply` / `scene.splat` — Gaussian splat (opens in SuperSplat, Postshot, etc.); for a video, the still
    background only
  - `scene.splatv` — videos: the whole 4D scene (background + everything that moves, frame by frame); plays in
    the app and in splaTV-compatible players
  - `scene.obj` (+ `.mtl`, texture) — mesh, when the backend provides one
  - `vr180/…_180_LR.jpg|mp4`, `…_180_TB.jpg|mp4` — VR180 stereo, with spherical metadata
  - `vr180/…_coverage.png` — white = seen, grey = interpolated, black = unknown
  - `README.txt` — explains every file
  - `_2d2vr180/` — metadata the app uses to reopen the scene (other programs ignore it)
  - `run_report.json` (one level up) — everything about the run

## How inputs are handled

| Input | What happens |
|---|---|
| Photo | Single-image 3D (SHARP if installed and allowed, otherwise MoGe-2, otherwise Depth-Anything). |
| Photo + AI views (Real 3D) | Qwen-Image-Edit draws the other angles (Wan 2.2 if Qwen is not installed); VGGT places every view in 3D, MoGe-3 (or MoGe-2) gives your photo's real-world scale and depth; one 3D splat is trained on the photo and the views, and its colours are finally taken from the view that sees each surface best. |
| Several photos | Multi-view engine: VGGT finds where each photo was taken, MoGe-2 gives real-world scale, a 3D Gaussian splat is trained on all photos. |
| Video (any) | **4D scene**: VGGT places every frame's camera, MoGe-2 gives every frame's depth at real-world scale; what moves is found by comparing each frame in 3D with frames before and after it; the still surroundings are trained once as one 3D splat from all frames (what someone hides in one frame is filled in from the others) and the moving parts become Gaussians that move smoothly between frames (frame by frame if that fails). Then a VR180 video is rendered from the 4D scene, from where the camera was (path smoothed and levelled). |
| Video, 4D engine not available | The sharpest frame is reconstructed as a photo (the report says why). |
| Video + Real 3D | Real 3D on the sharpest frame (for videos where nothing moves). |

Speed on an RTX 4080 (estimates): photo 20–60 s; several photos 5–15 min; video in 4D 8–25 min; generative 3D
10–25 min.

## Honesty about what is real

2D2VR180 can invent missing content when you ask it to, and it always tells you which parts are real:
- **Observed** geometry: seen in two or more of your photos/frames.
- **Inferred**: seen in one input image, depth predicted by a network.
- **Generative**: invented by the AI views (Real 3D, 360° capture, Explore). Plausible, not measured.
- **Interpolated**: gaps behind objects in VR180 filled from the background or by LaMa.
- **Unknown**: never seen — black in VR180.

Turn on *Colour by provenance* in the 3D viewer to see them (green observed, blue inferred, pink generative).
Without generative 3D a photo covers 60–80° of view, so a VR180 frame (180°×180°) is mostly black
around the picture; with *Explore* the surroundings are generated.

## Settings

- **Licence profile**: *Personal/research* allows every model; *Commercial* uses only models whose
  licence allows commercial use (MoGe-2, Depth-Anything-V2 Small).
- **VR180 renderer**: Automatic uses the GPU splat renderer when available, otherwise the CPU renderer.
- **Eye separation**, **resolutions**, **video length**, **hole filling** (and **AI hole filling** with LaMa),
  **CPU mode** (slow, for PCs without NVIDIA GPU; not available for multi-view or generative 3D),
  **copy results to** a folder, **Hugging Face token** (only for gated models).
- Data lives in `%LOCALAPPDATA%\2D2VR180`. Set `TWOD2VR180_HOME` to move it to another drive.

## VR: two ways to use a headset

**1. Walk around the 3D splat (6DoF) — PC-connected headset.** Connect the headset to this PC
(Quest Link cable, Air Link, Virtual Desktop or SteamVR). In **Results** (or the 3D viewer) press
**View in VR**: the scene opens in a WebXR viewer in Google Chrome or Microsoft Edge, served only to this computer
(`127.0.0.1`). Press **ENTER VR** at the bottom of the page. You start at eye level where the camera stood, with
the horizon levelled; move your
head to look around, or use the controllers: left stick = move, right stick = turn (left/right) and
rise/sink (up/down), A/X = back to the start. On the desktop: drag = orbit, right-drag = pan, wheel = zoom,
R = reset. From a single photo without generative 3D you will see empty space behind objects.
Command line: `2d2vr180-cli view-vr path\to\scene.ply`.

**2. Watch VR180 stereo — any headset, including standalone Quest.** Copy
`export\vr180\…_180_LR.mp4` (or `_180_TB.mp4`) to the headset (Quest: connect by USB → `Movies`) and play
it in the Files/Media app or any VR player (DeoVR, Skybox, Pigasus). The files carry VR180 metadata;
if the player asks, choose *180°* and *side-by-side* (or *top/bottom*).

**Other splat apps.** `scene.ply` is a standard 3D Gaussian Splatting file: it also opens in SuperSplat,
Postshot, Polycam, Luma and other splat viewers, some of which run on headsets.

## Troubleshooting

- **"No NVIDIA GPU detected"** — install/update the NVIDIA driver; check **Diagnostics**.
- **A component fails to install** — check free disk space and internet; press Install again
  (downloads resume). The error text is shown and kept in the app data folder.
- **A job fails** — the Results page shows the error; **Diagnostics → Copy to clipboard** gives a
  report you can attach to a GitHub issue. Nothing crashes the app: each backend runs in its own process.
- **Out of GPU memory** — use Fast mode or lower VR180 resolution in Settings.
