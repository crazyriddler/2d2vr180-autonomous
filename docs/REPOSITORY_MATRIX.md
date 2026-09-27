# Upstream Repository Matrix

Verified live on 2026-09-27 (commits in `config/upstream-lock.json`). "4080 16GB"
reflects documentation/code inspection only — no GPU test has run yet.

| Project | Role | Stage | Integration status | License (code / weights) | Category | Windows w/o toolchain | 4080 16GB |
|---|---|---|---|---|---|---|---|
| Splatline | orchestrator reference | P0 | reference only | MIT / — | reference-only | n/a | n/a |
| recon3d | video → 3DGS/mesh | P0 | adapter `recon3d_video`; Windows runtime install verified in CI (prebuilt gsplat 1.5.3+pt24cu124; no gtsam wheel → no factor-graph refinement) | MIT / VGGT-1B non-commercial | local-user-only | yes (verified) | likely (chunking) |
| VGGT | poses/depth | P0 | via recon3d | commercial-OK / 1B non-commercial, 1B-Commercial gated | redistributable-code-restricted-model | yes | likely with chunking |
| **MoGe** (new) | monocular metric geometry | P0 | adapter `moge_rgbd`; Windows runtime install verified in CI | MIT / MIT (HF cards) | redistributable | yes (verified) | yes (expected) |
| SHARP | single-image 3DGS | P1 | adapter `sharp` (research profile) | Apple / research-only | local-user-only | yes | likely |
| One2Scene | explorable single-image scene | P2 | unsupported | none / unspecified | blocked | no | unlikely (19 GB ckpt) |
| LongSplat | long-video 3DGS | P1 | unsupported | NVIDIA NC + Inria / MASt3R NC | local-user-only | no (CUDA ext.) | unknown |
| DepthSplat (`cvg/depthsplat`) | depth-conditioned 3DGS | P2 | not integrated | MIT / to verify | local-user-only | unknown | unknown |
| GSFixer | generative repair | P2 | unsupported | none / unspecified | blocked | no | unlikely |
| GSFix3D | repair/inpainting | P2 | unsupported | Apache+Inria (NC) / OpenRAIL++-M | local-user-only | no | unknown |
| gsplat | Gaussian rasteriser/training | P0 | inside runtimes | Apache-2.0 | redistributable | wheels for some combos | yes |
| **Depth-Anything-V2** (new) | relative depth | P1 | adapter `depth_anything_v2` (Small, via transformers) | Apache-2.0 / Small Apache-2.0 | redistributable (Small) | yes (verified) | yes |
| **Depth Pro** (new) | metric depth | candidate | not integrated | Apple / Apple | local-user-only | yes | likely |
| TripoSplat (new, via Splatline) | single-image 3DGS | candidate | not evaluated | reported MIT | to verify | to verify | to verify |
| FFmpeg 7.1 | media | P0 | bundled `bin/ffmpeg.exe` (separate process) | GPL-3.0-or-later | redistributable with source offer | yes | yes |

## License policy

A project can be `reference-only`, `local-user-only`, `redistributable`,
`redistributable-code-restricted-model` or `blocked`. Never infer model-weight
licensing from source-code licensing. Models are downloaded from their original
hosts by the model manager after the user accepts the license; nothing is mirrored.

## Additional research

Search for newer projects at the start of each phase; add only after license check,
reproducibility check, RTX 4080 test and quality comparison.
