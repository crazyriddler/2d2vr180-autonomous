# 2D2VR180 1.0.0-rc55 — Real 3D by Apple SHARP: the photo and 3 views given volume and joined

Owner verdict on rc54: still worse than rc19. Owner design for rc55:
- generate 3 and only 3 images, one attempt each, from the agreed angles (45° left, 45° right, from above);
- give the photo and each of those images volume with Apple SHARP;
- make the splat from those four;
- make it the default.

## How Real 3D works now

1. **3 views.** Qwen-Image-Edit draws 45° left, 45° right and from above, straight from the photo. There is
   one attempt per view in every mode: no candidates, no view from below.
2. **SHARP on four images.** Apple SHARP runs once over the photo and the 3 views. SHARP reads the lens from
   EXIF, so each image it gets carries the photo's real focal length. The app's working copy of the photo
   is a PNG without EXIF; without a lens, SHARP assumes 30 mm.
3. **Where each view was taken.** The camera engine (VGGT) poses the four images; only the poses are used,
   with no training. Each view's SHARP splat is scaled so its depth matches the camera engine's, in the
   photo's SHARP metres. Without the engine, the views start from the requested orbit around the subject,
   trying both turning directions.
4. **Laid onto the photo.** A coloured ICP fits a robust similarity, coarse to fine. Each view point is
   matched to the nearby photo surface point that is closest in space and colour. Colour stops a view
   sliding along flat surfaces. A fit that has to turn the camera more than 20° or resize the view more
   than 1.5× is rejected. A view that still does not line up with the photo is left out, with a warning.
5. **Only what the photo cannot show is added.**
   - The photo's splats are kept exactly as SHARP made them.
   - A view splat is dropped if the photo, or an earlier view, already has a surface there.
   - It is also dropped if it would sit in front of what the photo sees, or if it is one of SHARP's
     "rubber sheet" splats stretched across a depth edge of its own image.
   - What remains is the sides of the subject and what it hides, labelled GENERATIVE.
   - Each view's colours are matched to the photo, per-channel gain on the shared surfaces.
6. **The default.** Real 3D is the default preset, and the SHARP path is the default inside it. Installing
   Qwen-Image-Edit (the Real 3D component) now installs Apple SHARP too. Without SHARP, Real 3D falls back
   to the rc54 training path, with a log line saying so.

`run_report.json` records, per view:
- where its pose came from;
- the scale;
- how far the refinement turned it;
- how much of it lies on the photo's surfaces;
- how many splats it adds or drops, and why;
- its colour gain.

New code:
- `src/twod2vr180/fusion.py` (pure numpy, runs in the app);
- a `poses_npz` mode in `workers/multiview_worker.py`;
- `tests/unit/test_fusion.py`, with a synthetic box in front of a wall, views at the wrong scale, and both
  pose sources.

# 2D2VR180 1.0.0-rc54 — Real 3D: back to the front, faster and better than rc19

Owner verdict on rc53's Real 3D: worse and much slower than before. It did make a back, but:
- the back sat metres away from the subject;
- the subject looked distorted from every angle;
- parts of the back showed through the front.

The good version was rc19: good geometry, detail and scale when you don't turn your head too far. Owner
request: 3-4 AI views, a splat like rc19's but done properly, no back view, and fast.

## What went wrong and what was right

- **What rc19 got right.** Its three views (45° to each side and from above) are close to the photo, so the
  AI draws them consistently. The camera engine (VGGT) places them correctly, and geometry and scale come out
  right.
- **What rc19 got wrong: noisy texture.** The AI views do not quite agree with the photo in colour and detail.
  The training (spherical harmonics degree 3) memorised those disagreements, and the photo was trained at a
  low resolution.
- **What broke rc42-rc53: the invented back.**
  - The views at 90°, 135° and 180° were each drawn from the previous one, so they drifted. They no longer
    fitted the photo, so the camera engine put them metres away and their surfaces showed through the front.
  - TRELLIS.2's model of the whole subject did not fit the photo either.
  - Together with four rounds of drawing, all this made a job several times slower.

## What's new in rc54

- **Real 3D (the default) stays in front of the subject.** It draws 45° left, 45° right and from above; Auto
  and Quality add one from below. Every view is drawn straight from the photo, in one round.
- **Candidates are chosen inside the reconstruction.** In Auto/Quality, 2-3 candidates per view are drawn,
  and the reconstruction itself keeps the one that most exactly shows the photo from that angle, as in rc19.
  There is no separate scoring pass.
- **Kept from rc41-rc48**, everything that fixed the texture and the measurements:
  - spherical harmonics degree 1 (no memorised colour noise);
  - the photo trained at 1280 / 1600 / 2048 px;
  - colour correction of the AI views;
  - colours taken from the view that sees each surface best;
  - the camera's real focal length (EXIF);
  - real-world scale from the photo only;
  - robust depth alignment that leaves contradicting AI depth unguided;
  - MoGe-3 for the photo when it is installed.
- **Faster.**
  - One round of 3-4 views instead of four rounds of up to 8.
  - No TRELLIS.2 step.
  - Training at 5000 / 9000 / 15000 steps, as in rc19.
- **TRELLIS.2 removed.** Its engine, models and code are gone, so the Components page no longer offers them.
  If you installed it, you can delete the `subject3d-cu128` runtime folder and the `trellis*` model folders to
  free about 16 GB.
- *360° capture* still exists, for when you want the back and accept invented geometry.

## rc53: Memory check before every job

The owner tested rc52. Two failures happened right at the start of Real 3D, in steps that had worked in the
two runs before:

- TRELLIS.2: "CUDA out of memory" asking for 0.96 GB with 12.3 GB of VRAM free;
- loading Qwen's text encoder: the process crashed with Windows error 0xC0000005.

On Windows, GPU memory also needs "commit": room in RAM plus the page file. When that runs out, these are
exactly the symptoms. Free disk space had also dropped by 29 GB since the previous run, which fits a growing
page file. Something was holding a lot of memory before the job started: worker processes left over from
earlier failed jobs, or other programs.

## What's new in rc53

- **Leftover workers are stopped.** At the start of every job the app stops any of its own worker processes
  still running from earlier jobs. The queue runs one job at a time, so they are never needed. The log
  names each one and its memory.
- **Memory is reported.**
  - The log shows free RAM and free "memory for programs" (RAM + page file) before the job, and before
    every AI model step.
  - The report stores the same figures under `hardware.memory`.
- **A clear warning when memory is short.** With less than 24 GB free, the job says so, and what to do:
  close other programs, restart the PC if it stays low, and let Windows manage the page file on a drive with
  20 GB or more free.

## rc52: Real 3D memory fix, second try

The owner tested rc51. Real 3D now crashed at the first step of the first AI view, with Windows error
0xC0000005 (access violation) and no message. The mode rc51 introduced for PCs with 32 GB of RAM ("sync":
the AI image model's blocks copied to the GPU without streams) does not work with its quantised (GGUF)
weights on Windows.

## Fixes in rc52

- **Back to the streamed mode that worked**, with the real cause of the RAM exhaustion fixed. Each block of
  the 15 GB model is page-locked (pinned) on its way to the GPU. PyTorch keeps every freed pinned buffer in
  a cache and never gives it back to Windows, so the RAM filled up after two views. That cache is now
  emptied after every step and every view. The log shows how much pinned memory is still held after the
  first step.
- The "sync" mode is removed.
- Kept from rc51:
  - running out of RAM is retried with a lighter setting;
  - the VRAM limit follows the memory that is really free;
  - the photo given to Qwen is reduced to 2048 px.

## rc51: Real 3D no longer runs out of RAM

Owner report (rc50, Real 3D on a 24 MP photo, RTX 4080, 32 GB RAM): the job failed after the first two AI
views with `MemoryError: Unable to allocate 11.9 MiB`. The PC had run out of system memory (RAM), not GPU
memory.

## Fixes in rc51

- **Qwen-Image-Edit no longer pins its weights on PCs with less than 48 GB of RAM.** Its 15 GB model lives in
  RAM and moves to the GPU block by block.
  - The faster "stream" mode page-locks every block on the way, which on 32 GB left too little memory for
    everything else. Below 48 GB of RAM the blocks are now copied from ordinary memory: a little slower, with
    no extra RAM.
  - The log shows how much RAM is free and which mode is used.
- **Running out of RAM is retried, not fatal.** The views already drawn are kept, and the next, lighter
  setting runs in a fresh process. If nothing fits, the error says "out of system memory (RAM)" and suggests
  closing other programs or enlarging the Windows page file.
- **No more silent VRAM overflow.** Every GPU step limits itself to the VRAM that is really free, minus a
  margin.
  - Before, the limit was 92–94 % of the card's total. With the desktop and other programs using about
    1.5 GB, that was more than was free.
  - On Windows the driver then quietly backs the excess with system RAM: slower, and it eats the RAM.
  - Now a full card raises a normal out-of-memory error, which every step already handles by retrying
    smaller.
- **Big photos cost less.** The AI views are about 1 megapixel, so the photo given to Qwen is reduced to
  2048 px first instead of decoding all 24 MP. The 3D reconstruction still uses the photo at full detail.
- The log line "Local DINOv3 NOT found. Attempting download" comes from the TRELLIS.2 fork's code. It loads
  the model from the app's own folder, and that step runs with Hugging Face access switched off, so nothing
  is downloaded.

## rc50: Smooth movement in 4D videos

In rc49 the people and animals that move in a video were rebuilt frame by frame, like a flip-book: a new set
of splats every frame, with flicker between frames and nothing in between. Now they are **Gaussians that
move**.

## What's new in rc50

- **Moving parts that move.** The moving parts of every second frame are the starting point. Each group
  (a person, an animal) is followed to where it is in the next one: first as a whole, then point by point.
  Then every Gaussian learns how it moves (speed, acceleration, turning) and for how long it is visible. The
  optimisation uses every frame, in front of the trained background, and looks at:
  - the colour of the moving pixels;
  - their depth;
  - local rigidity: neighbours keep their distances, so a body does not tear apart;
  - smooth acceleration.
- **Smooth everywhere.**
  - The 4D player plays continuously, not frame by frame.
  - The VR180 video is rendered at 24 fps, with the camera and the movement interpolated between the
    video's frames, even when the video was sampled at 8–15 per second.
  - The 4D file is smaller, because the same Gaussians serve several frames.
- **Safe fallback.** If the motion optimisation fails (for example, out of memory), the moving parts are
  rebuilt frame by frame as in rc49, and the log says why.
- Still the side the camera saw. A person who turns round shows their other side only while it faces the
  camera.

## Not yet tested on your hardware

The motion optimisation was tested on the CPU on a synthetic moving object. Seeded from every second
frame, it halves the error of the frames in between and learns the right speed. The whole worker was run end
to end with stand-ins for the models. Speed and quality on real videos on the RTX 4080 are still to be
measured.

## rc49: Videos become 4D scenes

Until now a video only really used one frame. From this version every video becomes a **4D Gaussian splat
scene**: 3D that changes over time. It comes with a player for the app and for VR, and the VR180 video is
rendered from it.

## What's new in rc49

- **Videos in 4D.** For every frame of the video (by quality: 8, 12 or 15 per second, for 8, 12 or 15 s):
  - VGGT finds where the camera was, and MoGe-2 gives the depth at real-world scale. A camera on a tripod is
    detected and held exactly still.
  - **What moves is found in 3D.** Each frame is compared with frames before and after it and across the
    clip: if another frame sees empty space where a point is, or another colour at the same depth, that part
    is moving. Someone who stands still for a while and then walks away still counts as moving.
  - **The surroundings are trained once** as one 3D splat from all frames, leaving out what moves. What a
    passer-by hides in one frame is filled in from the frames where it is visible.
  - **What moves is rebuilt frame by frame** in the same 3D space, from the side the camera saw. It is
    labelled *inferred*.
- **4D player.**
  - **In the app**, the 3D viewer plays a 4D result with play/pause and a timeline.
  - **View in VR** opens a new 4D player in Chrome or Edge. You walk around the scene while it moves. The
    trigger plays and pauses; the sticks move and turn you as before.
- **New VR180 video.** It is rendered from the 4D scene: every frame from where the camera was, with the path
  smoothed and the horizon levelled, showing the background plus what moved at that moment. The old
  frame-by-frame video path, its 4D sequence of `.ply` files and the *Video* mode menu are gone.
- **New files.**
  - `scene.splatv` is the whole 4D scene. It uses splaTV's format, so it also plays in splaTV-compatible
    players.
  - `scene.ply` is the still background only, for any splat viewer.
- **New options** under *Several photos and videos*: *Video length (4D)* and *Video frames (4D)*. The
  recommended setting for both is "by quality". From the command line, use `--video-seconds`,
  `--video-fps` and `--video-start`.
- **If the 4D engine cannot run** (it needs the multi-view engine, VGGT, MoGe-2 and 10 GB of VRAM), the
  sharpest frame is reconstructed as a photo, and the report says why.

## Not yet tested on your hardware

The 4D engine has not been run on an RTX 4080 with the real models. These parts were tested without the real
models or a GPU:

- motion detection, on synthetic videos;
- the whole worker, run end to end with stand-ins for VGGT, MoGe-2 and gsplat;
- the players, in a browser and in the app.

Still to measure on your hardware: speed, memory, and how well motion is found in real footage (shadows,
reflections, lighting changes).

## Next

The moving parts are rebuilt per frame, so they only show the side the camera saw. True 4D Gaussians that
move and keep their volume (4DGT, Shape of Motion) are next. The `.splatv` format and the player already
support moving splats.

## rc48: MoGe-3 for your photo's geometry

Reviewing which method computes distances most precisely today on an RTX 4080 found one better than
MoGe-2: **MoGe-3** (Microsoft, MIT licence). Real 3D now uses it for your photo.

## What's new in rc48

- **MoGe-3 measures your photo.** It gives sharper depth, a better field of view and a better real-world
  scale than MoGe-2. In Real 3D, it now provides three things for your photo:
  - its depth, which guides the 3D shape;
  - its real-world size (the metric scale of the whole scene);
  - its field of view, when the photo has no EXIF focal length.
  - It runs in the photo engine (about 1.5 GB download, about 6 GB of VRAM, a few seconds). Install
    **MoGe-3 Large** in Models; it is recommended. Without it, Real 3D uses MoGe-2 as before. If MoGe-3 fails,
    the job continues with MoGe-2 and the log says why.
  - The report lists it in the models used.
- **Your camera's real lens now actually reaches the 3D.** rc47 meant to use the EXIF focal length, but the
  job works on PNG copies of your photo, which have no EXIF, so it was silently lost. The focal length is now
  read from the original file and travels with the photo through the whole pipeline.
- **Better focal length from EXIF.** The 35 mm-equivalent focal length is converted with the diagonal
  convention used by camera makers, after rotating the photo upright, and cropped photos are detected.

## Updating

The photo engine is reinstalled once (it gains the FlexGEMM and Triton kernels MoGe-3 needs).

## rc47: More precise distances

A review of how distances are computed for the 3D found four weak points. All four are fixed in this
version.

## What's new in rc47

- **One focal length for your photo.** Since rc41, MoGe-2 used your photo's real focal length (EXIF), but the
  cameras still used VGGT's own estimate. When the two differ, the shape stretches or flattens in depth. Now
  the EXIF value is used for the photo's camera as well.
  - It is ignored when it disagrees with the image by more than a third, because it is then probably wrong.
  - It is also ignored for cropped photos, detected because the shape no longer matches the original
    frame's.
  - The log shows how far VGGT's estimate was.
- **Real-world scale only from your photos.** The metric scale used to be measured on up to 4 images, AI views
  included. AI views have no real lens, so their metric depth is a guess. Now only the real photos count.
- **Each view's depth is aligned with scale and shift, robustly.** Before, MoGe-2's depth of each image was
  brought onto the joint geometry with one scale factor. Now a scale and a shift are fitted, and the pixels
  where the views disagree are rejected while fitting. In an AI view, the areas whose depth still contradicts
  what all views agree on are left unguided, instead of pulling the shape away.
  - In a synthetic test (a person 2 m in front of a wall), the old fit was up to 6.6 cm off; the new one is
    exact.
- **Your photo keeps guiding the depth until the end.** The depth guidance used to fade out completely at 60 %
  of the training, after which only colour shaped the 3D, and areas without texture (walls, smooth skin)
  could drift. Now your photo keeps a weak depth guidance to the end; the AI views still fade out as before.

## rc46: Cleanup: fewer engines, clearer menus

## What's new in rc46

- **Removed what the current workflow no longer uses.**
  - **Stable Virtual Camera** (engine, worker and 3 models, about 9 GB). It was only a fallback when Wan 2.2
    was missing, needed a Hugging Face token, and distorted people. Qwen-Image-Edit and Wan 2.2 cover every
    path. If you installed it, you can delete it from the models folder.
  - **"Sharp fusion" assembly.** It assembled views from each one's own MoGe-2 depth without training. Your
    tests found that worse than the trained splat, which is now the only way: Real 3D, 360° capture, Explore
    and several photos are all trained.
  - The "Around the subject", "Wide orbit" and "Spiral" paths, which overlapped with Real 3D and 360° capture.
- **Reorganised Create page.**
  - Three presets plus Custom: *Real 3D from one photo* (recommended), *Quick 3D*, *360° capture*.
  - Advanced options in three groups: *VR180 output*; *One photo* (AI views, AI model for the views,
    reconstruction engine); *Several photos and videos*.
- **Recommended components match the workflow.**
  - Qwen-Image-Edit and its parts are now recommended, because Real 3D needs them.
  - Wan 2.2 and Depth-Anything-V2 are optional: Wan is for exploring surroundings, and Depth-Anything-V2 only
    for the Commercial profile.
- **Real 3D without Qwen** now falls back to Wan 2.2's 360° capture instead of failing.
- Video is unchanged for now: it will be rebuilt on 4D Gaussian Splatting next.

## rc45: Optional: the complete subject in 3D with TRELLIS.2

## What's new in rc45

- **Complete subject (optional component "TRELLIS.2").** With it installed, Real 3D first builds the whole
  subject of your photo in 3D, back included, with Microsoft's TRELLIS.2. Only its shape is used:
  1. The mesh is placed exactly where the subject is in your photo. Its orientation comes from the
     silhouette, and its size and distance from MoGe-2's depth, refined by matching the two surfaces.
  2. In the AI views (sides, back), the subject's depth now comes from this mesh. Before, each view had its
     own monocular guess, and the guesses disagreed with each other. Something MoGe-2 sees clearly in front
     of the subject is kept.
  3. Its surface also joins the starting points of the splat, so the back has a shape from the start.

  Colours still come from your photo and the AI views. If TRELLIS.2 fails or is not installed, Real 3D
  works as in rc44.
- Install it under Components: "Complete-subject engine (TRELLIS.2)" plus "TRELLIS.2" (about 11 GB: only the
  shape models, not the texture ones). It runs in its own runtime, with the Windows builds of TRELLIS.2's
  CUDA extensions from the TRELLIS.2-stableprojectorz project.
- Fast uses TRELLIS.2's 512 pipeline; Auto and Quality use the 512 → 1024 cascade.
- Note: not yet run on the RTX 4080. The placement step is tested on synthetic scenes (millimetre accuracy
  for known orientations, about 4 cm on a 1.7 m subject tilted 10°).

## rc44: Each surface takes its colour from the view that sees it best

## What's new in rc44

- **Best-view colours.** Once the 3D shape is trained, a short final pass recalculates only the colours. Each
  surface's colour now comes from the view that sees it best: the most head-on view, with the most pixels
  on it. Your photo wins wherever it sees the surface.
  - Before, the colour was an average of the photo and the AI views. They never agree exactly, so the
    texture came out soft and noisy.
  - Nothing moves and nothing is removed: positions, sizes, shapes and opacities of every splat stay exactly
    as trained.
  - The log shows how much of each view ended up deciding the colours.
- Applies to every multi-view result with up to 24 images (Real 3D, several photos). Longer videos keep the
  normal training.

## rc43: The AI views' colour drift no longer stains the texture

## What's new in rc43

- **Per-view colour correction for the AI views.** Qwen's views usually come out a little brighter, darker
  or warmer than your photo. rc19 averaged those differences into the splat's colours, which looked blotchy
  where views overlap. Now each AI view gets a small colour correction while training (a 3x3 colour matrix
  plus an offset, kept close to "no change"). The splats keep your photo's colours, and each view's drift
  is absorbed by its own correction.
  - Only colours are affected: no splats are moved or removed.
  - The details panel shows the correction learned for each view.

## rc42: Real 3D all the way round: sides and back

## What's new in rc42

- **The back of your subject now exists.** rc19 never looked further than 45° to each side, so the back of
  a person had no image at all. Real 3D now draws the subject in rings around it:
  1. 45° left, 45° right and from above;
  2. 90° left and right;
  3. 135° left and right;
  4. the back (180°).

  Fast stops after the second ring (5 views); Auto and Quality draw all 8.
- **Each view continues its neighbour.** From the second ring on, every view is drawn from your photo *and*
  the neighbouring view already accepted (90° from 45°, 135° from 90°, the back from both 135° views), so
  clothes, hair and proportions stay the same all the way round.
- **Candidates are judged against their neighbour.** With 2–3 candidates per angle (Auto/Quality), the one
  that best continues its neighbouring view is kept. A back view shares nothing with your photo, so rc19's
  photo-based check would have rejected it. One candidate is still drawn from the photo alone, as a
  fallback.
- Training is unchanged from rc41.
- Generation takes longer: 4 rounds of Qwen instead of 1.

## rc41: rc19 plus cleaner textures and more accurate scale

Built on rc19 (rc40), following `docs/research/from-rc19.md`. Only "Real 3D from one photo" changes.

## What's new in rc41

- **Less texture noise.** The splat's colour now depends on the viewing angle only slightly (spherical
  harmonics degree 1 instead of 3). With 4 images, degree 3 let each splat show a different colour to each
  camera: the AI views' small disagreements were memorised as colour, which looked like noise and flicker.
- **Your photo's real detail.** rc19 shrank every image, the photo included, to 1024 px before training. Now the
  photo is trained at 1280 px (Fast), 1600 px (Auto) or 2048 px (Quality). The AI views stay at their native
  size (about 1 MP); enlarging them would add nothing.
- **More training steps for the larger photo:** 6000 (Fast), 12000 (Auto), 20000 (Quality).
- **More accurate scale and depth.** When the photo has EXIF data with the 35 mm-equivalent focal length (almost
  every phone and camera), MoGe-2 uses the real field of view instead of guessing it. A wrong field of view
  stretches or flattens the whole scene in depth. The log shows the EXIF value and VGGT's own estimate.
- Nothing else changes: the same views, cameras, depth priors and training as rc19, and no cleanup.

## rc40: Back to rc19

## What's new in rc40

- **Everything is back to how it was in rc19.** After many tests, rc19's "Real 3D from one photo" gave the
  best 3D geometry: Qwen-Image-Edit views, VGGT cameras, MoGe-2 depth priors and a trained splat.
- Every change from rc20 to rc39 is removed: Depth Anything 3, subject mode, the cleanup steps, colour
  correction, camera refinement, AbsGS, the feed-forward splat, FlashWorld, the photo layer, more angles,
  the layered mesh and the rest. The cleanup steps had removed detail from both the scene and the subject.
- The code is identical to rc19; only the version number changed. Improvements will be built again on this
  base, one at a time, starting from a research review (see `docs/research/`).

## rc39: A more complete 3D: more angles, and a mesh that is not set in stone

## What's new in rc39

- **Real 3D draws your subject from many more angles.**
  - Fast: 5 views (45° and 90° to each side, from above).
  - Auto: 9 views (adds 135° to each side, the back and from below).
  - Quality: 11 views (adds raised 45° diagonals).
  - Views of the back share almost nothing with your photo, so they are now judged by how far the camera
    moved. Before, they would have been dropped as "inconsistent" and retried.
- **The mesh (OBJ) is no longer "frozen in carbonite".**
  - The fusion joined the outline of the subject to the wall behind it with a continuous skin. Now the
    subject and the background are fused separately, and depth jumps are left out.
  - The subject is fused with much finer voxels: about 1.4 mm for a subject 0.84 m away, about 5× more
    detail than before.
  - `subject.obj`, the subject alone without the background, is also exported.
- **Mesh fusion is faster and always finishes.**
  - Since the mesh is made from the splat, the whole photo background was fused with the subject's small
    voxels. A background several times farther away then takes tens of times more voxels.
  - The background now uses coarser voxels.
  - A face budget caps every mesh: about 1.5 M faces for the subject and 0.5 M for the background. The
    voxel grows when that budget would be exceeded.

## rc38: The raw Depth Anything 3 splat is the result

## What's new in rc38

- **No more processing of the 3D.** In your tests, the raw Depth Anything 3 splat looked better than any
  trained, polished or fused version. So whenever Depth Anything 3 is installed and can take the images,
  its raw splat is the scene: the VR180 output, the PLY/SPLAT export and the turntable.
  - This applies to every multi-view job: Real 3D from one photo, 360° around the subject, and several
    photos of the same scene.
  - The splat is not trained afterwards, and no splats are removed in a cleanup step.
  - What rc37 added is kept, because it is part of how the raw splat is built, not processing on top: the
    photo layer (one splat per photo pixel) and the removal of duplicate copies.
  - The separate *Turntable: raw DA3 splat* button is no longer needed: *Play turntable* now shows it.
- Without Depth Anything 3, or with more than 32 images or images of different shapes (long videos, for
  example), the previous method is used.

## rc37: Real 3D: back to AI views, with your photo's full detail

## What's new in rc37

- **"Real 3D from one photo" uses Qwen-Image-Edit views again.** That was the route that gave the most real
  3D. FlashWorld is still available: choose it under *Generative engine*.
- **Your photo keeps its full detail.** Depth Anything 3 builds its splat at about 504 pixels, so each
  Gaussian covered 2–3 pixels of the photo, which looked blurry. Now the photo's part of the scene is one
  Gaussian per photo pixel (1280 px in Auto), with the photo's own colours. The depth comes from Depth
  Anything 3's joint estimate across all views, the smoother one that worked better. Pixels on depth edges
  are left out, so there are no streaks between the subject and what is behind it.
- **Fewer ghosts.** Every view used to add its own copy of each surface, each at a slightly different
  depth, which made the result look ghostly. Now a generated view only adds what the photo, or an earlier
  view, does not already show.
- **The polish no longer erases detail.** The splat used to be trained like one built from scratch: points
  moved, split and were pruned, which blurred it. Now only colours and opacities are adjusted. A ghost copy
  that the other views contradict fades out, and the geometry stays as built.
- **Local misalignments of AI views are absorbed.** Each AI-drawn view is usually a few pixels off here
  and there, for example a hand or an ear. Each view now gets a small, smooth image alignment while
  training, instead of the splat averaging the copies into blur. The log and the details panel show how
  far each view had to be shifted.
- The previous result (before the polish) is still exported as *Turntable: raw DA3 splat* for comparison.

# 2D2VR180 1.0.0-rc36 — FlashWorld fits a 16 GB card

## Fixed in rc36

- **FlashWorld ran out of GPU memory, then seemed to hang.**
  - FlashWorld passes all 24 views through its image encoder and 3D decoder at once. That needs about
    24 GB of VRAM.
  - The fallback then moved them to the CPU, which on Windows is so slow that it looked frozen: no GPU use,
    1 % CPU.
  - Now the views go through 4 at a time, then 1 at a time if memory is still short, always on the GPU.
    Each view passes through on its own, so the result is identical. A CPU run of the real engine gave a
    difference of exactly 0 between 1, 3 and all views at a time.
  - The CPU fallback is gone.
- The log now shows each of FlashWorld's 4 steps and the GPU memory peak, so progress is visible.

## Fixed in rc35

- **"Real 3D" with FlashWorld failed** with `AssertionError: torch.Size([1, 4])` as soon as generation
  started.
  - Cause: FlashWorld was written for a newer gsplat than 1.5.3, the version with ready-made Windows
    builds. In 1.5.3's default "packed" mode, per-view background colours are rejected.
  - FlashWorld now calls gsplat in unpacked mode. That changes only gsplat's internal memory layout, not
    the image.
  - A new test reproduces the exact error with the old call and passes with the new one.
- Everything before that point already worked on the RTX 4080: the text encoding (cached), MoGe-2
  measuring the subject at 0.84 m, and loading and quantising the 21 GB model in 22 seconds.

## What's new in rc34

- **A different strategy for "Real 3D from one photo".** Until rc33, an AI drew 3 more views of your photo
  one by one, and the app tried to rebuild a 3D from them. Each view was drawn on its own: a slightly
  different pose, hair and lantern. No reconstruction method can make one rigid 3D from that, so the
  result showed double faces and layers.
- **FlashWorld** (ICLR 2026) generates **the 3D itself**.
  - It is a Wan 2.2 video model with a 3D Gaussian decoder. At each of its 4 steps it builds the Gaussians,
    renders them and feeds them back into the model.
  - Every angle comes from the same 3D, so it cannot ghost between views.
  - The camera swings ±30° around your subject (±25° Fast, ±35° Quality) with a slight rise. Your photo is
    the starting point.
  - The result is placed in your photo's camera frame, at the subject's real distance (measured by MoGe-2).
- **Install it from Settings → Components → "FlashWorld — Real 3D generated directly from one photo".**
  - About 35 GB: the FlashWorld checkpoint is 20.9 GB, plus the Wan 2.2 VAE and text encoder (14 GB).
  - It runs in the multi-view engine. No new Python environment is needed, but the multi-view engine updates
    itself once (it adds diffusers and transformers).
  - FlashWorld's weights are CC BY-NC-SA 4.0 (non-commercial). The code is Apache-2.0.
  - Without it, "Real 3D" keeps using the previous flow.
- **Memory.** The 21 GB checkpoint is read straight from disk and quantised to FP8 layer by layer, so it
  never fills your RAM. The text encoder runs only once: its result is cached. On a 16 GB card the model
  leaves the GPU while the 3D decoder works. If memory still runs out, it retries automatically with the
  decoder on the CPU (slower).
- **Not tested on a GPU before release.** It was checked on the CPU with the real VAE architecture and a
  tiny transformer with random weights. Camera conventions were checked against FlashWorld's own code.
  The first run on your RTX 4080 is the real test: please send the log and the turntable.

## What's new in rc33

- **"Real 3D from one photo" no longer trains the splat from scratch.** Training a splat from only 4 views
  that do not match exactly (the AI changes small things in every view) produced double faces and torn
  hair. No amount of selecting, weighting or camera correcting removed that.
  - The splat now comes straight from **Depth Anything 3**. The Nested Giant-Large model you already have
    includes a head that predicts the 3D Gaussians from all views at once. It was trained to fuse views
    into one coherent 3D, and it does not invent ghosts where the views disagree.
  - In subject mode the person comes from that splat (all views), and the background from your photo.
  - A short, gentle polish follows to recover your photo's sharpness. Splats move 10× slower than in
    training from scratch, and there are 3000 steps in Auto (1500 Fast, 5000 Quality).
- **Compare both.** Results has a new button, **Turntable: raw DA3 splat**, which plays
  `turntable_feedforward.mp4`: the splat exactly as Depth Anything 3 gave it, before polishing. Its PLY is
  `export/scene_feedforward.ply`. "Play turntable" shows the polished final result.
- Without Depth Anything 3 (VGGT only), the old training is used.
- Colours of the feed-forward splat are view-independent: the extra Python package DA3 needs to rotate its
  view-dependent colours is not bundled. The polish adds view-dependence where your photo supports it.

## Fixed in rc32

- **Subject mode was only half applied.** In the rc31 test the subject was found in the photo but in none of
  the AI views. Depth Anything 3 then compared a subject-only photo with whole AI images, and its cameras
  went wrong (a "high angle" view measured at 79° instead of ~30°). The result was double faces and smeared
  hair. Now:
  - Once the photo shows a subject in front of a background, the AI views (the same scene) look for their
    near layer with looser criteria. A real depth jump is still required.
  - AI candidates without a subject are left out. They never sit next to subject-only views again. The log
    lists the result for every candidate.
- **Wrong angles count much more against a candidate.** A view measured far from the angle asked for (79°
  for a 30° view, or 6° for a 45° one) is not a clean camera move, so it rarely wins any more.

## Fixed in rc31

- **Subject mode.** With a studio portrait, Qwen turns the person ~45° but leaves the plain backdrop as
  it was. Depth Anything 3 then measured only 5–8° of camera movement, and training was torn between
  "the camera moved 45°" (the person) and "the camera barely moved" (the wall). The splat came out
  shredded. Now:
  - The person is separated from the background by depth. Subject mode switches on only when the photo
    clearly has two layers (a subject in front of a background), not for landscapes or rooms.
  - The cameras are found from the person alone.
  - AI views teach only the person. The background comes from your photo.
  - Results shows "subject mode" when it was used.
- **Joint multi-view depth instead of per-view MoGe-2 depth.** The sharper per-view MoGe-2 depth (since
  rc19) made the 3D worse than the earlier, smoother option. Each AI view got its own depth that disagreed
  with the others'. Training is back on the camera engine's depth (Depth Anything 3 / VGGT), which is
  estimated for all views together and is consistent.
- **The turntable swings around the subject, not around the wall behind it**, and only as far as the views
  reach. Before, it always swung 30°, which showed areas nobody had seen.

## Fixed in rc30

- **"Real 3D from one photo" failed** with `ImportError: cannot import name 'batch_align_poses_umeyama'
  from 'depth_anything_3.utils.pose_align'`.
  - Cause: the Giant / Nested Depth Anything 3 models build a Gaussian head that imports a pose-alignment
    helper from a module 2D2VR180 replaces, because that module's dependencies are not bundled. The
    replacement now serves every helper the model files import. Inference never calls them.
  - Checked by building the Nested Giant-Large model from its configuration (1.69 B parameters) and
    running a full inference on a portrait image.
- **If Depth Anything 3 cannot be loaded, the job continues with VGGT** and the log says why, instead of
  failing. This applies to both choosing the AI views and reconstructing.

## What's new in rc29

- **Finer texture in the 3D.** Training adds splats where detail is missing.
  - Before, that check summed the image error signals, and in fine texture (hair, fabric, foliage)
    opposite signals cancelled out, so those areas stayed blurry.
  - It now uses their absolute size (AbsGS, as in gsplat's reference trainer), so fine texture gets the
    splats it needs.
  - The splat limit per quality mode is unchanged. If the GPU runs out of memory, training retries smaller
    as before.

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
| `2D2VR180-1.0.0-rc36-setup.exe` | Installer — per user, no administrator rights |
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
