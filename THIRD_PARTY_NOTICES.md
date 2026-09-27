# Third-party notices — 2D2VR180

2D2VR180 itself is MIT-licensed (see LICENSE). The Windows package bundles or
downloads the components below. **No model weights are bundled.**

## Bundled in the installer / portable ZIP

| Component | Version | License | Notes |
|---|---|---|---|
| Python runtime (via PyInstaller) | 3.11 | PSF-2.0 | application interpreter |
| NumPy | 2.4.6 | BSD-3-Clause | |
| Pillow | 12.3.0 | MIT-CMU (HPND) | |
| Qt for Python (PySide6-Essentials, shiboken6) | 6.11.2 | LGPL-3.0 | dynamically linked, unmodified Qt libraries in the install folder may be replaced by the user |
| uv | 0.12.19 | MIT OR Apache-2.0 | bin/uv.exe — installs the backend runtimes |
| FFmpeg | 7.1 (imageio-ffmpeg 0.6.0 build) | **GPL-3.0-or-later** | bin/ffmpeg.exe, run as a separate process. Source: https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz ; configuration: `ffmpeg -buildconf`. |

## Downloaded on demand into %LOCALAPPDATA%\2D2VR180 (never bundled)

Runtimes (config/runtime-manifest.json): PyTorch (BSD-3-Clause, bundles NVIDIA CUDA
runtime libraries under the NVIDIA EULA), torchvision, gsplat (Apache-2.0), MoGe
(MIT), utils3d-moge, pipeline, transformers (Apache-2.0), Apple ml-sharp (Apple Sample Code License), recon3d
(MIT), VGGT (VGGT License v1), GTSAM (BSD), Open3D (MIT) and their dependencies.

Models (config/model-manifest.json) — shown with their license before download:

| Model | License | Commercial use |
|---|---|---|
| MoGe-2 ViT-S / ViT-L (normal) | MIT (Hugging Face model cards, verified 2026-09-27) | yes |
| Depth-Anything-V2 Small | Apache-2.0 (Hugging Face model card) | yes |
| Apple SHARP | Apple ML Research Model License — research only | **no** |
| VGGT-1B (fetched by recon3d) | non-commercial | **no** |

The default license profile is "Personal / research use". The "Commercial use"
profile disables every backend whose model license does not permit commercial use
(SHARP and the recon3d/VGGT video engine); MoGe-2 and Depth-Anything-V2 Small remain available.
