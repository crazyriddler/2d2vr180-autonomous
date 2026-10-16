# 2D2VR — User guide

2D2VR turns **photos into 3D Gaussian splats** and **videos into splat videos** that you walk into in VR.
Everything runs on your own computer: nothing you process is uploaded anywhere.

## Requirements

- Windows 10 (build 19041+) or Windows 11, 64-bit
- An NVIDIA GPU with a recent driver (target: RTX 4080 16 GB; 8 GB+ works). Driver 570 or newer.
- 32 GB of RAM recommended (the AI painter of photos needs it)
- About 60 GB of free disk space for the engines and models, plus space for your results
  (a splat video takes 2-5 GB per minute)
- Internet access **only** for the one-time download of the engines and models

You do **not** need Python, CUDA, Git, Conda, Visual Studio or anything else.

## Install

1. Download `2D2VR-<version>-setup.exe` from the GitHub release (or the portable ZIP).
2. Run it. It installs for your user only; no administrator rights are needed.
   - It replaces 2D2VR180 1.x if it is installed.
   - Your downloaded models and results are kept: the data folder `%LOCALAPPDATA%\2D2VR180` is renamed to
     `%LOCALAPPDATA%\2D2VR` on first start.
3. Start **2D2VR** from the Start menu.

## First start: install the components

On first start the app offers to install the recommended components. **Install recommended** installs
everything Real 3D uses; you do not need to pick anything.

| Component | What it does | Size |
|---|---|---|
| Real 3D engine | runs the models below on your GPU | ~5.5 GB |
| Apple SHARP | the 3D Gaussians of every photo and video frame (**research licence**) | ~2.5 GB |
| MoGe-3 Large | measured depth, lens and real-world scale | 1.5 GB |
| YuNet face finder | the eyes of the people, for the real-world size | 0.2 MB |
| Sapiens2 body parts + 3D shape of people | real noses and faces instead of flat ones | 1.6 + 6.5 GB |
| AI painter engine + Qwen-Image-Edit-2511 | photos: paints what the photo hides when you move your head | ~6 + ~30 GB |

Every model shows its licence before downloading. Downloads resume if interrupted and are verified
(SHA256) before use.

**Coming from 1.x?** The Components page lists the models and engines 1.x downloaded that 2.0 no longer uses
(VGGT, MoGe-2, Wan 2.2, LaMa, the multi-view engine…) with a **Delete them** button: up to ~45 GB back.

## Make something

1. Open **Create** and drop photos or videos (or click to browse). Each file becomes its own result.
2. **Quality**: *Auto* (recommended), *Quality* (more detail; for videos more splats per frame, slower) or
   *Fast*.
3. **Eye distance (IPD)**: the real-world size of the scene is set from the eyes of the people in it, scaled
   to be this far apart (adults 56-70 mm, 63 on average).
   - If people look too big in VR, lower it; too small, raise it.
   - In VR you can also resize everything with the controller buttons (below).
4. **Photos: AI paints what the photo hides** (on by default): the photo's splat is looked at from a little to
   each side, and what the photo never saw (behind edges) is painted by Qwen-Image-Edit and added at a depth
   tied to the photo's. These parts are labelled *generative*.
5. **Videos → splat video**:
   - *Start at* and *Length* choose the part of the video.
   - *Frames per second*: by default the video's own rate, up to 24.
6. Press **Generate**. Jobs run one after the other; the log shows every step.

### What Real 3D does with a photo

- Uses your photo's real lens (EXIF) and its measured metric depth (MoGe-3).
- Gives people their real 3D shape (Sapiens2, a model trained only on humans).
- Sets the real-world size from the people's eyes (or from the human model when there are no faces).
- Aligns Apple SHARP's 3D Gaussians of the photo to that depth: SHARP's sharpness with true geometry.
- Paints what the photo hides from moved positions (optional) and adds only those parts.

### What a splat video is

Every frame of the video made the same way (without the AI painting, which would take minutes per frame),
in one file with the video's sound:

- The lens and the real-world size are measured once for the whole clip, so the scene does not "breathe".
- Each frame's depth is matched to the previous frame's where the picture did not change, which removes the
  small jumps of depth estimated one frame at a time.
- **It is slow but steady**: about 3-5 s per frame on an RTX 4080.
  - A minute of video at 24 fps takes 1.5-2 hours.
  - Memory use stays the same however long the video is: frames are written to disk as they are made.
  - If the GPU runs out of memory, the job continues from the last finished frame with lighter settings.

## Results

**Results** lists every job. For each one:

- **Play / explore in 3D** opens it in the app's 3D viewer: drag to orbit, right-drag to pan, wheel to zoom.
  - A splat video plays with its sound; the timeline under the view seeks.
  - When a frame is not ready yet, playback waits instead of skipping, so the speed stays steady.
- **View in VR** opens it in the VR player (next section).
- **Open folder** shows the files; each result folder has a `README.txt` explaining them:

| File | What it is |
|---|---|
| `scene.splv` | the splat video, with the sound (videos) |
| `scene.ply` | the splat (photos) or the first frame (videos), standard 3DGS format |
| `scene.splat` | the same in the compact web format |
| `views_sheet.png` | the splat seen from a few moved positions (photos) |
| `people_shape.png` | the people's relief: the scene's depth, with their own shape, and the splats as made |
| `added_layer.ply`, `fill_steps.png` | what the AI painted, and how (photos) |
| `compare_sharp_alone.ply` | SHARP without the measured depth, for comparison (photos) |
| `../run_report.json` | every detail of the run: models, timings, scale, warnings |

## View in VR

The player opens in **Google Chrome** or **Microsoft Edge** on this PC. It is served only to this computer
(127.0.0.1); nothing is uploaded.

1. Connect your headset to the PC: **Quest Link** (cable), **Air Link**, **Virtual Desktop** or **SteamVR**.
2. Press **ENTER VR** in the page. You start where the camera stood, with the horizon level.
3. Controllers:

| Button | Action |
|---|---|
| **A** or **X** (hold) | make everything smaller |
| **B** or **Y** (hold) | make everything bigger |
| Stick click | back to the start, at the original size |
| Trigger | play / pause a splat video |
| Left stick | move |
| Right stick | turn 30° (left/right), rise/sink (up/down) |

- The size you choose is remembered for that scene.
- The app's 3D viewer also has a **Size in VR** box to set the starting size.
- Without a headset the page works with the mouse; **+** and **−** change the size, **Space** plays/pauses.

## Settings

- **Licence profile**: *Personal / research use* (default) or *Commercial use*. Real 3D needs Apple SHARP,
  whose licence is research-only, so it does not run in the Commercial profile.
- **CPU mode**: lets the engines run without an NVIDIA GPU (extremely slow; for testing only).
- **Also copy results to**: a folder where every result is copied when it is done.
- **Data folder**: set the environment variable `TWOD2VR_HOME` before starting the app to use another drive.

## Command line

`2d2vr-cli.exe` does everything the app does:

```
2d2vr-cli doctor                      # GPU / driver / FFmpeg
2d2vr-cli backends                    # what can run here, and why not
2d2vr-cli run photo.jpg --ipd-mm 60   # a photo's Real 3D splat
2d2vr-cli run clip.mp4 --video-seconds 20 --video-fps 15
2d2vr-cli view-vr result\export\scene.splv --size 0.9
```

## Troubleshooting

- **A job fails**: the message says why; Diagnostics and the job's `run_report.json` have the details.
- **Out of GPU memory**: close other GPU programs (games, browsers with many tabs). Splat videos continue
  with lighter settings by themselves.
- **"ENTER VR" is missing**: use Chrome or Edge, and start Quest Link / SteamVR before opening the page.
- **People look too big or too small**: lower or raise the eye distance and make it again, or resize with
  A/B in VR.
