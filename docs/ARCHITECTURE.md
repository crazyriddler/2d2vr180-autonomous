# Architecture (2D2VR 2.0)

```text
   GUI (PySide6)            CLI
   Create · Results · 3D viewer · Components · Settings · Diagnostics
        │                    │
        └────────┬───────────┘
                 ▼
            Job runner (jobs.py) ── run_report.json for every job
    ingest → select → reconstruct → validate → export (+ README.txt)
                 │
     ┌───────────┴─────────────────────────────┐
     ▼                                         ▼
 photo_locked (backends/locked.py)       splat_video (backends/video_splat.py)
  1 MoGe-3 (+ Depth Pro): lens + depth     frames (FFmpeg, JPEG) + sound (Opus/WebM)
  2 Sapiens2: people's shape               workers/video_splat_worker.py, models loaded once:
  3 scale from the eyes (YuNet)              calibration: one lens + one scale per clip
  4 normals → fine relief, clean edges       per frame: MoGe-3 → Sapiens2 → scale →
    SHARP with the depth, every splat placed,
    fitted to the photo (refine.py)              depth stabilised to the previous frame →
  5 AI painter (painter.py → Qwen)
    paints the holes of moved views              SHARP with depth → fixed grid subset →
  6 SHARP on the painted views,                  quantize + filters + deflate → frames/N.bin
    added only in the holes (fusion.py)    assemble scene.splv (splv_codec.write_file)
     │                                         │
     ▼                                         ▼
 scene.ply / scene.splat                  scene.splv (+ frame 0 as scene.ply / .splat)
     │                                         │
     └──────────────┬──────────────────────────┘
                    ▼
   App 3D viewer (gui/viewer.py, gl_view.py)       Browser VR player (assets/webxr/player.*)
   OpenGL point sprites; .splv: decoder thread,     served on 127.0.0.1 (vr_server.py, HTTP range);
   sound (QtMultimedia) as the clock                WebGL2 splat rasteriser, decode worker pool,
                                                    sort worker, <audio> as the clock, WebXR
```

## Design rules

- The GUI never imports research code. Each upstream project is used by a worker script that runs inside an
  isolated runtime (`photo-cu128`: MoGe, SHARP, Depth Pro, Sapiens2; `render-cu124`: gsplat with its prebuilt CUDA
  kernels - the renders of moved views and the fit to the photo; `gen-cu128`: Qwen-Image-Edit) and talks
  JSON lines on stdout (`workers/_protocol.py`). The app kills the worker's process tree on cancel.
- Every splat carries its provenance: INFERRED (seen in the photo/frame, depth predicted) or GENERATIVE
  (painted by AI). Painted parts are exported separately (`added_layer.ply`).
- Fail honestly: a backend that cannot run says why (selection → `rejected` in the report); a GPU out of
  memory in a splat video resumes from the last finished frame with lighter settings, and the report says so.
- Memory is bounded everywhere: the video worker writes each frame to disk as soon as it is made; the file is
  assembled by streaming; both players decode a few seconds ahead at most.

## The splat video format (.splv)

Defined in `workers/splv_codec.py` (numpy + zlib only, shared by the workers and the app; the browser player
decodes the same layout):

- `b"SPLV"`, version, a JSON header (fps, frames, splats per frame, quantization ranges, the filming camera,
  the frame index with keyframe flags, the audio track), then the data.
- Every frame has the same K splats in the same order: a fixed subset of SHARP's pixel-aligned grid. So an
  order sorted for one frame is valid for the next, and still parts of the scene are identical across frames.
- 17 bytes per splat before compression:
  - position as image-plane u, v and log depth (16 bits each);
  - log size relative to the depth (3 × 8 bits);
  - rotation quaternion (4 × 8 bits);
  - colour and opacity (4 × 8 bits).
- Per field and frame the encoder picks a filter (difference to the previous frame and/or to the previous
  splat), splits 16-bit values into byte planes, and deflates. Keyframes every second allow seeking.
- The sound is stored whole (Opus in WebM, else AAC) after the frames.

## Precision of a photo's 3D (2.0.0-rc7)

`workers/precision.py` (numpy + torch), used by `moge_worker.py` and `sharp_depth_worker.py`:

| Step | Function | What it fixes |
|---|---|---|
| lens check | `lens_choice` | EXIF vs MoGe-3's and Depth Pro's own field of view; theirs when both disagree with it |
| several passes | `combine_passes` | MoGe-3 on the photo, its mirror image and a coarser level, median in log depth |
| second depth | `depth_ensemble` | Apple Depth Pro joined: overall shape from both, edges from Depth Pro |
| normals → relief | `align_normals`, `integrate_normals` | MoGe-3's and Sapiens2's normals turned into the photo's axes (orthogonal Procrustes) and joined as slopes of log depth (screened Poisson, CG), capped |
| clean edges | `resolve_edges` | pixels next to a depth step go to the near or far side by colour |
| placement | `sharp_depth_worker.snap_to_depth` | every seen splat at that depth; at edges cell-centre sampling and the back layer on the far side |
| fit | `workers/refine.py`, `refine_worker.py` | colours, opacities, sizes fitted to the photo from its camera (gsplat, `render-cu124`), positions fixed |

Maximum detail (`workers/detail_geometry.py`) does the same at up to 3072 px (4608 px, 3 x 3 parts, for
photos of 24 MP and up), with each part's normals turned into the photo's axes first.

## Real-world scale

- MoGe-3 gives metric depth.
- The eyes of the people found by YuNet are brought to the eye distance chosen by the user (63 mm by
  default); without faces, Sapiens2's distance of the people sets it.
- In a splat video the scale is measured once on frames spread over the clip.
- In VR, A/X and B/Y scale the whole scene about the camera's place, to correct what is still off.

## Files

| Module | Role |
|---|---|
| `jobs.py` | the job pipeline and its report |
| `selector.py` | photo → `photo_locked`, video → `splat_video`, with the reasons when they cannot run |
| `backends/locked.py`, `backends/painter.py`, `fusion.py` | Real 3D for photos |
| `workers/precision.py`, `workers/refine.py`, `workers/detail_geometry.py` | precision and detail of photos |
| `backends/video_splat.py`, `workers/video_splat_worker.py` | splat videos |
| `splv.py`, `workers/splv_codec.py` | the splat video format |
| `vr_server.py`, `assets/webxr/player.*` | the browser VR player |
| `gui/viewer.py`, `gui/gl_view.py` | the app's 3D viewer |
| `models.py`, `runtimes.py`, `components.py` | downloads, verification, the Components page |
| `media.py` | FFmpeg: probing, frames, sound, EXIF lens |
