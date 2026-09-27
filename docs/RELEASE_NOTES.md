# 2D2VR180 0.1.0 — engineering preview (NOT RELEASED)

This version has **not** passed the mandatory GPU and licensing gates
(`docs/status/gates.json`) and must not be published as a user release.

## What works (verified without a GPU)
- Desktop app: drag-and-drop photo/video, Auto/Quality/Fast, Generate/Cancel, progress/log,
  3D viewer with provenance colouring, VR180 preview, diagnostics, model & runtime manager.
- Video analysis that avoids redundant frames (fixed camera + static scene → one frame).
- Gaussian splat `.ply`/`.splat` and textured OBJ export; import of third-party 3DGS PLY (incl. SHARP).
- VR180 SBS/TB stills and H.264 MP4 rendered from the 3D scene, with coverage masks and honest FOV reporting.
- `run_report.json` for every job, including failures and cancellations.

## Not yet verified
- Backend inference on an RTX 4080 (MoGe-2, SHARP, recon3d).
- Runtime installation on Windows; clean-machine installer test.

## Hardware support matrix
| GPU | Status |
|---|---|
| NVIDIA RTX 4080 16 GB | target — untested |
| Other NVIDIA ≥ 8 GB, driver ≥ 570 | expected to work for photo backends — untested |
| No NVIDIA GPU | app runs; viewer/VR180 of existing splat files only |
