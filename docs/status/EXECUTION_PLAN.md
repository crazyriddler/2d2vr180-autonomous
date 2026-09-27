# Execution plan and status

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

## Milestone 3 — video
- ✅ Video analysis: fixed vs moving camera (phase correlation), moving subjects, scene cuts, keyframes, redundancy
- ✅ Static scene → single sharpest frame; fixed camera + motion → per-frame 2.5D; moving camera → multi-view
- 🟡 `recon3d_video` adapter (VGGT + MoGe-2 + gsplat) — runtime unverified on Windows

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
- ✅ Linux frozen build smoke-tested; 🟡 Windows build runs in CI (`.github/workflows/build.yml`)
- ⬜ Clean Windows VM install test

## Milestones 8–9 — generative repair, specialists
- ⛔ GSFixer / GSFix3D / One2Scene evaluated and registered as unsupported with reasons (licence / toolchain / VRAM)

## Release
Blocked by mandatory gates A, B, C-photo-inference-gpu, G (see `docs/status/gates.json`).
No GitHub Release has been published.
