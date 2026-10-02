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
| three.js | 0.170.0 | MIT | assets/webxr/three.module.js — WebXR splat viewer |
| GaussianSplats3D (@mkkellogg/gaussian-splats-3d) | 0.4.7 | MIT | assets/webxr/gaussian-splats-3d.module.js — WebXR splat viewer |
| uv | 0.12.19 | MIT OR Apache-2.0 | bin/uv.exe — installs the backend runtimes |
| FFmpeg | 7.1 (imageio-ffmpeg 0.6.0 build) | **GPL-3.0-or-later** | bin/ffmpeg.exe, run as a separate process. Source: https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz ; configuration: `ffmpeg -buildconf`. |

## Downloaded on demand into %LOCALAPPDATA%\2D2VR180 (never bundled)

Runtimes (config/runtime-manifest.json): PyTorch (BSD-3-Clause, bundles NVIDIA CUDA
runtime libraries under the NVIDIA EULA), torchvision, gsplat (Apache-2.0), MoGe
(MIT), utils3d-moge, pipeline, transformers (Apache-2.0), Apple ml-sharp (Apple Sample Code License), recon3d
(MIT), VGGT (VGGT License v1), Open3D (MIT), scikit-learn (BSD-3-Clause), Stability AI stable-virtual-camera
(Stability AI Non-Commercial Research Community License), diffusers (Apache-2.0), open_clip (MIT),
kornia (Apache-2.0), roma (BSD-3-Clause), VideoX-Fun (Apache-2.0), transformers (Apache-2.0), omegaconf (BSD-3-Clause),
librosa (ISC) and their dependencies.

Models (config/model-manifest.json) — shown with their license before download:

| Model | License | Commercial use |
|---|---|---|
| MoGe-2 ViT-S / ViT-L (normal) | MIT (Hugging Face model cards, verified 2026-09-27) | yes |
| Depth-Anything-V2 Small | Apache-2.0 (Hugging Face model card) | yes |
| Apple SHARP | Apple ML Research Model License — research only | **no** |
| VGGT-1B | CC-BY-NC-4.0 (Hugging Face model card) | **no** |
| Wan 2.2 Fun 5B Control-Camera (alibaba-pai) | Apache-2.0 | yes |
| Stable Virtual Camera 1.1 | Stability AI Non-Commercial Research Community License (gated; outputs non-commercial) | **no** |
| Stable Diffusion 2.1 VAE (sd2-community mirror) | CreativeML Open RAIL++-M | yes (use restrictions apply) |
| OpenCLIP ViT-H/14 LAION-2B | MIT | yes |
| LaMa (big-lama) | Apache-2.0 | yes |

The default license profile is "Personal / research use". The "Commercial use"
profile disables every backend whose model license does not permit commercial use
(SHARP, the VGGT multi-view engine and Stable Virtual Camera); MoGe-2, Depth-Anything-V2 Small and LaMa
remain available.
