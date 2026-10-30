# 2D2VR — User guide

2D2VR turns **photos into 3D Gaussian splats** and **videos into splat videos** that you walk into in VR.
Everything runs on your own computer: nothing you process is uploaded anywhere.

## Requirements

- Windows 10 (build 19041+) or Windows 11, 64-bit
- An NVIDIA GPU with a recent driver (target: RTX 4080 16 GB; 8 GB+ works). Driver 570 or newer.
- 16 GB of RAM (32 GB recommended for Quality on very large photos)
- About 40 GB of free disk space for the engines and models, plus space for your results
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
| Real 3D renderer | draws the splat on the GPU (gsplat): the fit to the photo (Auto, Quality), the views sheet | ~3 GB |
| Apple SHARP | the 3D Gaussians of every photo and video frame (**research licence**) | ~2.5 GB |
| MoGe-3 Large | measured depth, lens and real-world scale | 1.5 GB |
| MoGe-3 Giant | photos in Auto and Quality: the most accurate depth | 5 GB |
| Apple Depth Pro | photos in Auto and Quality: a second measured depth, sharp edges, lens check (**research licence**) | 1.9 GB |
| YuNet face finder | the eyes of the people, for the real-world size | 0.2 MB |
| Sapiens2 body parts + 3D shape of people | real noses and faces instead of flat ones | 1.6 + 6.5 GB |
| Sapiens2 surface normals of people | photos in Auto and Quality: folds of clothes, fingers, the relief of a face | 6.2 GB |

Every model shows its licence before downloading. Downloads resume if interrupted and are verified
(SHA256) before use.

**Old downloads.** The Components page lists the models and engines this version no longer uses with a
**Delete them** button:
- from 2.0.0-rc8 and before: the AI painter engine and Qwen-Image-Edit, about 36 GB;
- from 1.x: VGGT, MoGe-2, Wan 2.2, LaMa and the multi-view engine, up to about 45 GB.

## Make something

1. Open **Create** and drop photos or videos (or click to browse). Each file becomes its own result.
2. **Quality** (the box under it says what each one does):

   | | Photos | Videos |
   |---|---|---|
   | **Fast** (~45 s a photo, ~2 s a frame) | MoGe-3 depth once, people's own 3D shape, clean depth edges, every splat at the measured depth | ~330 000 splats per frame, depth at 768 px; people's shape, clean edges and every splat placed, as in every mode |
   | **Auto** (~2 min, ~3 s a frame, recommended) | + MoGe-3 Giant three times joined to Apple Depth Pro (truer shape and lens), fine relief from surface normals (faces, fingers, folds of clothes), the splat fitted to the photo (sharper texture) | ~470 000 splats per frame, depth at 1024 px: sharper |
   | **Quality** (~5 min, ~5 s a frame) | + maximum detail: the photo worked at 3072 px (4608 px from 24 MP), the depth and SHARP on parts of it, 2.5-6x the splats (heavier in VR); a longer, finer fit to the photo | ~740 000 splats per frame, depth at 1536 px, + fine relief from surface normals |

   Times are for an RTX 4080 and a 24 MP photo or a 1080p video.
3. **Eye distance (IPD)**: the real-world size of the scene is set from the eyes of the people in it, scaled
   to be this far apart (adults 56-70 mm, 63 on average).
   - If people look too big in VR, lower it; too small, raise it.
   - In VR you can also resize everything with the controller buttons (below).
4. **Videos → splat video**:
   - *Start at* and *Length* choose the part of the video.
   - *Frames per second*: by default the video's own rate, up to 24.
5. Press **Generate**. Jobs run one after the other; the log shows every step.

### What Real 3D does with a photo

Every mode:
- Uses your photo's real lens (EXIF), checked against what the depth models see: a zoomed or cropped photo
  whose EXIF still names the full lens is caught.
- Measures its metric depth with MoGe-3.
- Gives people their real 3D shape (Sapiens2, a model trained only on humans).
- Sends every pixel next to a depth edge to the near or the far surface, so no splats float in between.
- Sets the real-world size from the people's eyes (or from the human model when there are no faces).
- Places Apple SHARP's 3D Gaussians of the photo at that depth: SHARP's sharpness with true geometry.

Auto and Quality also:
- Measure the depth with MoGe-3 Giant three times (the photo, its mirror image, a coarser pass) and with a
  second model, Apple Depth Pro, and join them (Depth Pro draws the sharpest edges).
- Add the fine relief the depth models smooth away, from surface normals (MoGe-3's; Sapiens2's for people).
- Fit the splat to the photo seen from its own camera (colours, opacities, sizes; nothing moves).

Nothing the photo does not show is invented: what is hidden behind people and objects stays empty.

### What a splat video is

Every frame of the video made like a photo in Fast (the slower photo steps would take minutes per frame),
in one file with the video's sound:

- The lens and the real-world size are measured once for the whole clip, so the scene does not "breathe".
- Each frame's depth is matched to the previous frame's where the picture did not change, which removes the
  small jumps of depth estimated one frame at a time.
- Pixels next to a depth edge go to the person or to the background, never in between, so no splats float
  between them (in every mode).
- **It is slow but steady**: about 2-5 s per frame on an RTX 4080, by quality mode.
  - A minute of video at 24 fps takes 50 minutes (Fast) to 2 hours (Quality).
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
| `relief.png` | the photo's depth at each step, shaded: MoGe-3, + Depth Pro, + people, final (normals, edges) |
| `compare_normal_detail.ply` | Quality: the same photo without maximum detail, for comparison |
| `compare_sharp_alone.ply` | Auto and Quality: SHARP without the measured depth, for comparison (photos) |
| `../run_report.json` | every detail of the run: models, timings, scale, warnings |

## View in VR

The player opens in **Google Chrome** or **Microsoft Edge** on this PC. It is served only to this computer
(127.0.0.1); nothing is uploaded.

1. Connect your headset to the PC: **Quest Link** (cable), **Air Link**, **Virtual Desktop** or **SteamVR**.
2. Press **ENTER VR** in the page. You start at the **ideal view**:
   - where the camera stood (the only place where nothing is missing behind people and objects);
   - level, facing what it photographed or filmed, at your own eye height;
   - one step back if the subject is closer than 0.9 m, which is uncomfortable in a headset.
3. Controllers:

| Button | Action |
|---|---|
| **A** or **X** (hold) | make everything smaller |
| **B** or **Y** (hold) | make everything bigger |
| **Click either stick** (left or right) | back to the ideal view, from wherever you have walked or turned to; the size stays |
| **A + B** (or **X + Y**) together | the original size |
| Trigger | play / pause a splat video |
| Left stick | move |
| Right stick | turn 30° (left/right), rise/sink (up/down) |

- The size you choose is remembered for that scene.
- The app's 3D viewer also has a **Size in VR** box to set the starting size.
- Without a headset the page works with the mouse: it starts at the same ideal view and turns around the
  subject; **R** goes back to it, **+** and **−** change the size, **Space** plays/pauses.

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
