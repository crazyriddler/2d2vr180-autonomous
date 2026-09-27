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

On first start the app offers to install the recommended components. You can also open
**Components** at any time.

| Component | What it enables | Size |
|---|---|---|
| Photo & fixed-camera video engine | required for every reconstruction | ~5.5 GB |
| MoGe-2 Large | Quality mode (metric depth, textured mesh) | 1.3 GB |
| MoGe-2 Small | Fast mode | 0.14 GB |
| Depth-Anything-V2 Small | fallback depth model (Apache-2.0) | 0.1 GB |
| Apple SHARP *(optional)* | best single-photo 3D; **research / non-commercial licence** | ~2.5 GB |
| Moving-camera video engine *(optional)* | multi-view video reconstruction (recon3d + VGGT) | ~7 GB + models on first use |

Every model shows its licence before downloading. Downloads resume if interrupted and are
verified (SHA256) before use.

## Make something

1. Open **Create** and drop one or more photos or videos (or click to browse).
2. Choose **Auto** (recommended), **Quality** or **Fast**.
3. Choose outputs: VR180 side-by-side and/or top/bottom, VR180 or flat 3D projection.
4. Click **Generate**. Jobs run one after another; you can cancel any of them.

When a job finishes, **Results** shows it. From there you can:
- **Explore in 3D** — GPU viewer: orbit (drag), pan (right-drag), zoom (wheel), splat size, colour by provenance.
- **View in VR (6DoF)** — look around the 3D splat in a headset connected to this PC (see below).
- **Open VR180 image / Play VR180 video**.
- **Open folder** — the `export` folder contains:
  - `scene.ply` / `scene.splat` — Gaussian splat (opens in SuperSplat, Postshot, etc.)
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
| Video, fixed camera, nothing moves | Treated as a photo: the sharpest frame is used (no wasted time on identical frames). |
| Video, fixed camera, people/animals move | Each frame is reconstructed on its own → a VR180 video of the motion. |
| Video, moving camera | Multi-view reconstruction (moving-camera engine). Without it, the sharpest frame is used and you are told. |

## Honesty about what is real

A single photo cannot show what is behind objects. 2D2VR180 never pretends otherwise:
- **Inferred** geometry: seen in the input, depth predicted by a network.
- **Observed** geometry: reconstructed from several video frames.
- **Interpolated**: small gaps next to objects filled from the background (can be turned off).
- **Unknown**: never seen — black in VR180.

A photo usually covers 60–80° of view, so a VR180 frame (180°×180°) is mostly black around the
picture. The app says so in the Results page and in `run_report.json` instead of inventing content.

## Settings

- **Licence profile**: *Personal/research* allows every model; *Commercial* uses only models whose
  licence allows commercial use (MoGe-2, Depth-Anything-V2 Small).
- **VR180 renderer**: Automatic uses the GPU splat renderer when available, otherwise the CPU renderer.
- **Eye separation**, **resolutions**, **video length**, **hole filling**, **CPU mode** (slow, for PCs
  without NVIDIA GPU), **copy results to** a folder, **Hugging Face token** (only for gated models).
- Data lives in `%LOCALAPPDATA%\2D2VR180`. Set `TWOD2VR180_HOME` to move it to another drive.

## VR: two ways to use a headset

**1. Walk around the 3D splat (6DoF) — PC-connected headset.** Connect the headset to this PC
(Quest Link cable, Air Link, Virtual Desktop or SteamVR). In **Results** (or the 3D viewer) press
**View in VR**: the scene opens in a WebXR viewer in Microsoft Edge, served only to this computer
(`127.0.0.1`). Press **ENTER VR** at the bottom of the page. You start where the camera stood; move your
head to look around. From a single photo you will see empty space behind objects — nothing is invented.
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
