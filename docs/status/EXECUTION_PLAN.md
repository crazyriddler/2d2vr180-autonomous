# Execution plan and status

## 2.0 — the reform (2026-10-14): Real 3D photos and splat videos, VR180 removed

- ✅ Rename 2D2VR180 → 2D2VR (package `twod2vr`, data folder migrated on first start, installer replaces 1.x)
- ✅ Removed: VR180 renderer and outputs, 4D video (video4d, scene4d, .splatv), multi-view, Quick 3D, rc54/rc55 Real 3D
  methods, three.js viewer; models VGGT, MoGe-2, Depth Anything, Wan 2.2, LaMa, angles LoRA; runtime recon3d-cu124
- ✅ Components: only what Real 3D uses, all recommended; old downloads listed with a delete button
- 🟡 `splat_video` backend + `workers/video_splat_worker.py`: Real 3D per frame (MoGe-3, Sapiens2, eye scale, SHARP with
  depth), one lens and scale per clip, depth stabilisation, resumable frames, OOM → lighter settings. Exercised with
  stand-in models; real models not yet run on the RTX 4080
- ✅ `.splv` format (`workers/splv_codec.py`): fixed splat subset, 17 B/splat quantized, per-field temporal/spatial
  filters, byte planes, deflate, keyframes, Opus sound
- ✅ Browser player (`assets/webxr/player.*`): .splat and .splv, range streaming, decode worker pool, sound as the clock,
  WebXR size buttons (A/X, B/Y, stick click), trigger play/pause — headless Chromium tests
- ✅ App viewer plays .splv with sound (QtMultimedia) and a timeline; decoder thread; GL and CPU paths tested
- ⬜ Owner test on the RTX 4080 + headset

## Earlier (1.x)


Legend: ✅ done and tested here · 🟡 implemented, not verified on target hardware · ⛔ blocked · ⬜ not started

## Milestone 1 — diagnostics, skeleton, job system
- ✅ `twod2vr180.hardware` (nvidia-smi probe, RAM, disk, FFmpeg) + `2d2vr180-cli doctor`
- ✅ `tools/diagnostics/gpu_smoke_test.py` (runs inside a runtime) — 🟡 not run on a GPU
- ✅ Job system (`jobs.py`): stages, progress events, cancellation (kills worker tree), `run_report.json` for every outcome
- ✅ Backend contract (`backends/base.py`): prepare / can_run / estimate / run / export / cleanup + availability reasons
- ✅ Worker protocol (`workers/_protocol.py`), isolated runtimes via bundled `uv` (`runtimes.py`)

## Milestone 2 — photo backend
- 🟡 `moge_rgbd`: MoGe-2 point map → surface-aligned Gaussians + textured OBJ (conversion tested; model never run)
- 🟡 `sharp`: Apple SHARP adapter (research profile only)
- 🟡 Real 3D v3 (rc56, default): the photo locked — SHARP aligned to MoGe-3 metric depth at the real lens,
  eye-distance scale check, A/B export + views sheet (`backends/locked.py`, `workers/sharp_depth_worker.py`).
  rc57: holes of moved views (what the photo never saw) painted by Qwen/LaMa, depth tied to the scene,
  SHARP with composite depth, only hole splats added (`fusion.unseen_by_photo`, `fusion.add_fills`).
- 🟡 rc58: scale always calibrated to the eye distance (setting, default 63 mm; human-model fallback); people's
  3D shape from Sapiens2 pointmap + seg fused into MoGe depth (`workers/human_geometry.py`, vendored backbone)
- 🟡 Real 3D AI views joined (rc55, now an option): 3 Qwen views (one attempt each) + SHARP on the photo and each view, joined by
  `fusion.py` (VGGT poses or requested orbit, coloured ICP, photo-first dedup, rubber-sheet filter, colour gain).
  Fusion unit-tested on synthetic scenes; job path tested with fake workers; not yet run on the RTX 4080

## Milestone 3 — video (rebuilt on 4D in rc49)
- ✅ Video analysis: fixed vs moving camera (phase correlation), moving subjects, scene cuts, best frame
- ✅ 4D scene model (`scene4d.py`): background + time-local / moving splats, `.splatv` read/write (splaTV layout)
- ✅ 4D players: WebXR/desktop `player4d` (browser test, SwiftShader) and the app viewer's timeline (offscreen + Xvfb GL)
- 🟡 `video4d` worker: VGGT poses for every frame, MoGe-2 aligned depth, moving masks by 3D reprojection (unit-tested on
  synthetic scenes), masked background training, per-frame moving splats — dry-run end to end with mocked models;
  never run with the real models / GPU
- 🟡 VR180 video from the 4D scene (GPU renderer draws the splats of each frame's time); old per-frame and
  camera-path VR180 code removed
- 🟡 Moving parts as moving Gaussians (rc50, Spacetime-Gaussian optimisation seeded by group following; CPU-tested on
  synthetic motion), continuous playback and 24 fps VR180
- ⬜ Real volume for moving subjects (Shape of Motion-style motion bases + 2D tracks)

## Milestone 4 — viewer / export
- ✅ PLY (INRIA 3DGS layout) + `.splat` + OBJ/MTL/texture; provenance sidecar; SHARP-style PLY import
- ✅ Interactive viewer (orbit/pan/zoom, provenance colouring) in the PySide6 GUI

## Milestone 5 — VR180
- ✅ Stereo from the 3D scene (never pixel shifting): parallel eyes, half-equirect VR180 and flat 3D, SBS/TB
- ✅ Coverage masks, content-FOV measurement, `is_full_vr180` + honesty note
- ✅ H.264/yuv420p MP4 encode with `_180_LR`/`_180_TB` naming and Spherical Video V2 `st3d`/`sv3d` metadata (VR180 equi bounds), verified by FFmpeg

## Milestone 6 — model manager
- ✅ Manifest, license acceptance, resumable `.part` downloads, SHA256 pin or TOFU, verify, delete, disk check
- ⛔ Model SHA256/size/revision values (Hugging Face unreachable from sandbox)

## Milestone 7 — installer
- ✅ PyInstaller spec (GUI + CLI exes, bundled ffmpeg/uv), Inno Setup per-user installer, portable ZIP, checksums, release manifest
- ✅ Windows CI (`.github/workflows/build.yml`): tests, PyInstaller build, frozen acceptance, installer + ZIP, silent install / run without Python / uninstall
- ⬜ Clean Windows 11 VM install test with interactive GUI launch

## Milestones 8–9 — generative repair, specialists
- ⛔ GSFixer / GSFix3D / One2Scene evaluated and registered as unsupported with reasons (licence / toolchain / VRAM)

## Release
Blocked by mandatory gates A (clean VM), B, C-photo-inference-gpu, G (see `docs/status/gates.json`).
No GitHub Release has been published.
