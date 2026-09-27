# Bootstrap Status — first autonomous pass

## Date
2026-09-27

## Environment used for this pass
Linux container (4 CPU, 16 GB RAM, **no GPU**). Network policy allowed GitHub git
operations and PyPI only; `huggingface.co`, `ml-site.cdn-apple.com`,
`download.pytorch.org`, `api.github.com` and GitHub release downloads were blocked
(HTTP 403 from the egress proxy). Consequently **no model was downloaded and no
inference was run**. Every GPU claim below is marked as not yet verified.

## Upstream commits
All ten planned repositories plus three new candidates were resolved with
`git ls-remote` and shallow-cloned for inspection; exact commits are in
`config/upstream-lock.json` (`python scripts/resolve_upstreams.py` reports drift).

- `DongLiangSXU/DepthSplat` does **not** resolve; the official repository is `cvg/depthsplat`.
- New: `microsoft/MoGe` (monocular metric geometry, MIT code), `DepthAnything/Depth-Anything-V2`,
  `apple/ml-depth-pro`. Splatline's README also points to TripoSplat (MIT) as a commercial-safe
  single-image 3DGS option — to evaluate next.

## License findings
| Project | Code | Weights | Category |
|---|---|---|---|
| recon3d | MIT | uses VGGT-1B (non-commercial) | local-user-only |
| VGGT | VGGT License v1 (commercial OK) | VGGT-1B non-commercial; VGGT-1B-Commercial gated | code-ok, restricted model |
| MoGe | MIT | **unverified** | local-user-only until verified |
| SHARP | Apple sample code | Apple ML Research Model License: research only | local-user-only |
| One2Scene | **no LICENSE** | unspecified | blocked |
| LongSplat | NVIDIA non-commercial + Inria | MASt3R (CC-BY-NC) | local-user-only |
| DepthSplat | MIT | to verify | local-user-only |
| GSFixer | **no LICENSE** | unspecified | blocked |
| GSFix3D | Apache-2.0 + Inria (non-commercial) | OpenRAIL++-M | local-user-only |
| gsplat | Apache-2.0 | — | redistributable |
| Splatline | MIT | — | reference-only |

Consequence: every currently integrable reconstruction backend is non-commercial or
unverified. The app therefore defaults to a "Personal / research use" license profile
and a "Commercial use" profile that disables them (tested).

## Windows findings (from code/README inspection)
- MoGe-2 v2 inference and SHARP prediction are pure PyTorch → no compiler needed.
- gsplat needs a prebuilt Windows wheel for the chosen torch/CUDA pair or it JIT-compiles
  (MSVC + CUDA toolkit) — this decides whether recon3d can ship.
- LongSplat, GSFixer, GSFix3D require compiled CUDA extensions (simple-knn,
  diff-gaussian-rasterization) → excluded under the "no toolchain" rule.
- One2Scene: Linux scripts, ~19 GB checkpoint, no license → excluded.

## RTX 4080 findings
Not measured (no GPU). Planned budgets: MoGe-2 ≤ 6 GB, SHARP ≤ 8 GB (estimate),
recon3d chunk size 20 on 16 GB (upstream: ~40 frames per VGGT batch on 24 GB).

## Model sizes
Unknown for MoGe-2 (HF API unreachable); SHARP ~2.5 GB (Splatline README);
One2Scene denoise ~19 GB + scaffold ~1.9 GB (upstream README).
`scripts/resolve_models.py --write` fills sizes/SHA256/revisions on a networked machine.

## Backend benchmark
None yet. CPU reference renderer (4 cores): 786k splats → 2048² VR180 stereo pair in 3.5 s, 1280² in 2.1 s;
source-view PSNR on synthetic scenes > 25 dB (test-enforced).

## What was built in this pass
See `docs/status/EXECUTION_PLAN.md`.

## Blockers
1. RTX 4080 machine (or cloud CUDA GPU) with unrestricted network for runtime install + inference.
2. MoGe-2 weight license verification (Gate G).
3. Windows gsplat wheel availability for recon3d.

## Next actions
1. On a GPU machine: `2d2vr180-cli runtimes install photo-cu128`, run
   `tools/diagnostics/gpu_smoke_test.py`, download MoGe-2, run a test JPG, record VRAM/timing.
2. `python scripts/resolve_models.py --write` and review model licenses.
3. Evaluate TripoSplat (MIT) and Depth-Anything-V2-Small (Apache-2.0) as commercial-safe photo paths.
4. Windows CI build → clean-VM install test → update `docs/status/gates.json`.
