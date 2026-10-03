# 2D2VR180 — User guide

2D2VR180 turns photos and videos into explorable 3D scenes (Gaussian splats and meshes)
and stereoscopic VR180 images and videos. Everything runs on your own computer: nothing you
process is uploaded anywhere.

## Requirements

- Windows 10 (build 19041+) or Windows 11, 64-bit
- An NVIDIA GPU with a recent driver (target: RTX 4080 16 GB; 8 GB+ works for photos;
  12 GB+ for moving-camera video). Driver 570 or newer is recommended.
- About 8 GB of free disk space for the engine and models (+ space for your results)
- Internet access **only** for the one-time download of the engine and models

You do **not** need Python, CUDA, Git, Conda, Visual Studio or anything else.

## Install

1. Download `2D2VR180-<version>-setup.exe` from the GitHub release (or the portable ZIP).
2. Run it. It installs for your user only; no administrator rights are needed.
3. Start **2D2VR180** from the Start menu.

The portable ZIP works the same way: unzip anywhere and run `2D2VR180.exe`.

## First start: install the components

On first start the app offers to install the recommended components (about 40 GB in total).
You can also open **Components** at any time and install only what you need.

| Component | What it enables | Size |
|---|---|---|
| Photo engine | every single-photo reconstruction, fixed-camera video, GPU VR180 renderer | ~5.5 GB |
| MoGe-2 Large / Small | metric depth for photos and video; real-world scale for the multi-view engine | 1.3 / 0.14 GB |
| Depth-Anything-V2 Small | fallback depth model (Apache-2.0, commercial OK) | 0.1 GB |
| Apple SHARP | best direct single-photo 3D (**research licence**) | ~2.5 GB |
| Multi-view engine + VGGT-1B | several photos of one place, videos where the camera moves, Generative 3D (**non-commercial**) | ~7 GB + 5 GB |
| Generative engine | Stable Virtual Camera and LaMa | ~5.5 GB |
| LaMa | AI filling of the gaps behind objects in VR180 | 0.2 GB |
| Wan 2.2 Fun 5B Control-Camera | generative 3D: films new camera moves around your photo (Apache-2.0; keeps people intact) | ~25 GB (needs ~32 GB RAM) |
| Qwen-Image-Edit-2511 + Multiple-Angles + Lightning *(optional)* | generative 3D: redraws your photo from other camera angles, sharp ~1 MP stills (Apache-2.0) | ~33 GB (needs ~32 GB RAM) |
| Stable Virtual Camera 1.1 *(optional)* | alternative generative engine for scenes/objects without people (**non-commercial, gated**) | 5 GB + 0.3 GB + 3.9 GB |

Stable Virtual Camera is optional and needs a free Hugging Face account (accept the licence at
https://huggingface.co/stabilityai/stable-virtual-camera, create a *Read* token, paste it in Settings).

Every model shows its licence before downloading. Downloads resume if interrupted and are
verified (SHA256) before use.

## Make something

1. Open **Create** and drop one or more photos or videos (or click to browse).
2. Answer **What do you want to make?**
   - **Real 3D from one photo (recommended).**
     - **With FlashWorld installed** (Components, ~35 GB): the whole 3D scene is generated directly from your
       photo, as the camera swings around the subject. Every angle comes from the same 3D, so there are no
       double contours. Takes a few minutes on an RTX 4080.
     - **Without FlashWorld:** AI (Qwen-Image-Edit + Multiple-Angles) makes 3 more views of your photo: 45°
       left, 45° right and from above.
     - In Auto and Quality, 2-3 candidates are made per angle, and the one that best behaves like a pure camera
       move of your photo is kept automatically. Views that contradict the photo are dropped.
     - The multi-view engine (Depth Anything 3 if installed, otherwise VGGT) then finds where each view was
       taken and trains a 3D splat from the four. That gives real volume and real-world scale.
     - **Results → AI views overview** shows every candidate, its score and which one was used.
     - **Results → Play turntable** plays a short video of the 3D swinging left and right, so you can judge its
       volume without the headset.
   - **Quick 3D.** Depth from the photo alone (MoGe-2 / SHARP). Takes seconds, but it is a relief rather than
     full volume.
   - **360° around the subject (experimental).** The sides, back, top and bottom are invented and joined with
     sharp fusion.
   - **Custom.** Every setting is under *Advanced options* (below).
3. Choose **Quality**: Auto (recommended), Quality (slower) or Fast. Tick **Also make VR180** for VR180
   stills and videos.
4. **Advanced options** (folded away):
   - **Photo: generative 3D** (Custom) — *Off* uses only what the photo shows. *3 views (Qwen)* makes 45° left, 45° right
     and a high-angle view of your photo (in Auto/Quality 2-3 candidates per angle, the one most consistent
     with a pure camera move is kept automatically) and reconstructs them with the photo in the multi-view
     engine (trained splat). Choosing the *Multi-view* backend with one photo does the same. *360° photo capture* makes a few widely
     spaced views starting from your photo: 45°, 90°, 135°, 180° (the back is invented), 270°, overhead and
     from below; in Quality mode it adds 225°, 315° and four raised diagonals. *Around the subject* films four short
     camera moves (45° to each side, from above, from below) and is the best choice for people; *Wide orbit* also
     invents the back (objects), *Explore* invents the surroundings so a VR180 is filled (best for VR180),
     *Spiral* adds a small, faithful extension.
   - **Generative engine** — *Qwen-Image-Edit* (with the Multiple-Angles LoRA) draws every angle as a sharp
     ~1 megapixel image; *Wan 2.2* films the camera moving around the subject and takes the frames at those
     angles (smoother, 704 px); *Automatic* uses Qwen when it is installed, then Wan 2.2.
   - **Generative 3D assembly** — *Sharp fusion* (default) gives every view MoGe-2's crisp per-pixel geometry,
     keeps the photo exactly as it is and adds only what the other views show (sides, top, underside): as
     sharp as a single photo from any angle. *Trained splat* fits one model to every generated frame: smoother
     transitions but softer, because the generated frames never agree perfectly.
   - **Video** — *Automatic* analyses the video; *Whole video → one 3D scene* reconstructs every part of the
     video into one splat (camera must move); *Frame by frame* for people/animals moving in front of a fixed
     camera; *Sharpest frame only*.
   - **Combine several photos into one 3D scene** — photos of the same place taken from different
     positions are reconstructed together (the more angles, the more complete).
   - **Also save one 3D splat per frame** — a 4D sequence for fixed-camera videos with movement.
5. Click **Generate**. Jobs run one after another; you can cancel any of them.

When a job finishes, **Results** shows it. From there you can:
- **Explore in 3D** — GPU viewer: orbit (drag), pan (right-drag), zoom (wheel), splat size, colour by provenance.
- **View in VR (6DoF)** — look around the 3D splat in a headset connected to this PC (see below).
- **Open VR180 image / Play VR180 video**.
- **Open folder** — the `export` folder contains:
  - `scene.ply` / `scene.splat` — Gaussian splat (opens in SuperSplat, Postshot, etc.)
  - `scene.obj` (+ `.mtl`, texture) — mesh, when the backend provides one
  - `vr180/…_180_LR.jpg|mp4`, `…_180_TB.jpg|mp4` — VR180 stereo, with spherical metadata
  - `vr180/…_coverage.png` — white = seen, grey = interpolated, black = unknown
  - `sequence/frame_*.ply` — one splat per video frame (when enabled)
  - `README.txt` — explains every file
  - `_2d2vr180/` — metadata the app uses to reopen the scene (other programs ignore it)
  - `run_report.json` (one level up) — everything about the run

## How inputs are handled

| Input | What happens |
|---|---|
| Photo | Single-image 3D (SHARP if installed and allowed, otherwise MoGe-2, otherwise Depth-Anything). |
| Photo + generative 3D | Wan 2.2 films 1–4 short camera moves that start at your photo (Stable Virtual Camera if Wan is not installed). Sharp fusion: VGGT places a few key views in 3D, MoGe-2 gives each one sharp geometry, and the photo is kept whole while the key views add only what it does not show. Trained splat: one 3D splat optimised on all frames. |
| Several photos | Multi-view engine: VGGT finds where each photo was taken, MoGe-2 gives real-world scale, a 3D Gaussian splat is trained on all photos. |
| Video, camera moves | Multi-view engine on up to 80 keyframes of the whole video. |
| Video, fixed camera, people/animals move | Each frame is reconstructed (depth stabilised over time) → a VR180 video of the motion, optional 4D sequence. |
| Video, fixed camera, nothing moves | The sharpest frame is used (or choose *Whole video → one 3D scene*). |

Speed on an RTX 4080 (estimates): photo 20–60 s; several photos / video 5–15 min; generative 3D 10–25 min.

## Honesty about what is real

2D2VR180 can invent missing content when you ask it to, and it always tells you which parts are real:
- **Observed** geometry: seen in two or more of your photos/frames.
- **Inferred**: seen in one input image, depth predicted by a network.
- **Generative**: invented by the generative engine (only when *Photo: generative 3D* is on). Plausible, not measured.
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
  **copy results to** a folder, **Hugging Face token** (needed for Stable Virtual Camera).
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
