# Upstream Repository Matrix

This file is a research plan. Claude must verify each repository live before using it.

| Project | Role | Stage | Expected use | License risk | 4080 16GB |
|---|---|---|---|---|---|
| Splatline | integration reference/orchestrator | P0 | photo/video 3DGS | inspect dependencies | likely |
| recon3d | video reconstruction reference | P0 | video → 3DGS/mesh | inspect | likely |
| VGGT | geometry/poses/depth | P0 | multi-frame and single-view geometry | checkpoint license must be verified | likely with chunking |
| One2Scene | single image → explorable scene | P1 | novel-view scene | inspect model license | unknown until benchmark |
| SHARP | single-image 3DGS | P1 | fast photo → 3DGS | checkpoint is reported as non-commercial; do not redistribute blindly | likely |
| LongSplat | long-video 3DGS | P1 | temporal coherent scene | inspect CUDA/build | benchmark |
| DepthSplat | depth-conditioned 3DGS | P2 | multi-view reconstruction | inspect | benchmark |
| GSFixer | generative novel-view repair | P2 | fill artifacts/unseen regions | model licenses + heavy runtime | likely difficult |
| GSFix3D | novel-view repair | P2 | repair/inpainting | inspect | benchmark |
| gsplat | rendering/training library | P0 | Gaussian renderer/training | inspect | likely |
| FFmpeg | media pipeline | P0 | decode/encode | LGPL/GPL build choice must be deliberate | yes |

## Additional research

Claude must search GitHub for better or newer projects at the start of each major phase. New projects may be added only after:
- license check;
- reproducibility check;
- RTX 4080 test;
- quality comparison.

## License policy

A project can be:
- `reference-only`;
- `local-user-only`;
- `redistributable`;
- `redistributable-code-restricted-model`;
- `blocked`.

Never infer model-weight licensing from source-code licensing.

The application may download a model from its original host at runtime when its terms permit that use, but the model must not be mirrored into the Git repository or release asset unless redistribution is explicitly permitted.
