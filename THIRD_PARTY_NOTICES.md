# Third-party notices — 2D2VR

2D2VR itself is MIT-licensed (see LICENSE). The Windows package bundles or
downloads the components below. **No model weights are bundled.**

## Bundled in the installer / portable ZIP

| Component | Version | License | Notes |
|---|---|---|---|
| Python runtime (via PyInstaller) | 3.11 | PSF-2.0 | application interpreter |
| NumPy | 2.4.6 | BSD-3-Clause | |
| Pillow | 12.3.0 | MIT-CMU (HPND) | |
| Qt for Python (PySide6-Essentials, shiboken6) | 6.11.2 | LGPL-3.0 | dynamically linked, unmodified Qt libraries in the install folder may be replaced by the user |
| splaTV / splat (antimatter15, Kevin Kwok) | main, 2024 | MIT | assets/webxr/player.js — the player's splat rasteriser (shaders, instanced quads, worker depth sort) adapted for .splat and .splv; LICENSE-splatv.txt |
| Sapiens2 backbone (facebookresearch/sapiens2) | see config/upstream-lock.json | Sapiens2 License (workers/LICENSE-sapiens2.md) | workers/sapiens2_backbone.py, vendored verbatim; runs only inside the downloaded runtime |
| uv | 0.12.19 | MIT OR Apache-2.0 | bin/uv.exe — installs the backend runtimes |
| FFmpeg | 7.1 (imageio-ffmpeg 0.6.0 build) | **GPL-3.0-or-later** | bin/ffmpeg.exe, run as a separate process. Source: https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz ; configuration: `ffmpeg -buildconf`. |

## Downloaded on demand into %LOCALAPPDATA%\2D2VR (never bundled)

Runtimes (config/runtime-manifest.json): PyTorch (BSD-3-Clause, bundles NVIDIA CUDA runtime libraries under
the NVIDIA EULA), torchvision, gsplat (Apache-2.0), MoGe (MIT), utils3d-moge, pipeline, FlexGEMM (MIT), Apple
ml-sharp (Apple Sample Code License), Apple ml-depth-pro (Apple Sample Code License), transformers (Apache-2.0),
safetensors (Apache-2.0), OpenCV (Apache-2.0), triton-windows (MIT), timm (Apache-2.0) and their dependencies.

Models (config/model-manifest.json) — shown with their license before download:

| Model | License | Commercial use |
|---|---|---|
| Apple SHARP | Apple ML Research Model License — research only | **no** |
| MoGe-3 ViT-L and ViT-G (Microsoft) | MIT | yes |
| Apple Depth Pro | Apple ML Research Model License — research only | **no** |
| YuNet face detector (OpenCV Zoo) | MIT | yes |
| Sapiens2 seg 0.4B, pointmap 1B and normal 1B (Meta) | Sapiens2 License | see licence |

The default license profile is "Personal / research use". The "Commercial use" profile disables every
backend whose model license does not permit commercial use; Real 3D needs Apple SHARP, so it is not available
there.
