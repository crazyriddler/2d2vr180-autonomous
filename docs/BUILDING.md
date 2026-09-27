# Building 2D2VR180 (reproducible)

## Windows release build (what CI runs)

Requirements on the build machine only: Windows 10/11 x64, Python 3.11, PowerShell,
Inno Setup 6 (installed automatically via Chocolatey if missing). End users need none of these.

```powershell
./scripts/bootstrap.ps1   # venv from requirements-dev.txt (pinned) + SHA256-verified uv.exe / ffmpeg.exe
./scripts/build.ps1       # PyInstaller -> dist\2D2VR180 (2D2VR180.exe GUI, 2d2vr180-cli.exe) + frozen smoke test
./scripts/test.ps1        # pytest + scripts/acceptance.py against the frozen CLI
./scripts/package.ps1     # release\: portable ZIP, setup EXE, RELEASE_NOTES, checksums.sha256, release-manifest.json
./scripts/release.ps1     # refuses unless docs/status/gates.json mandatory gates pass
```

Pinned inputs: `requirements-app.txt`, `requirements-dev.txt`, `config/build-tools.lock.json`
(uv 0.12.19, FFmpeg 7.1 from imageio-ffmpeg 0.6.0, both SHA256-verified),
`config/runtime-manifest.json` (per-backend wheels), `config/upstream-lock.json` (upstream commits).
`release-manifest.json` records all of these plus the source commit and asset hashes.

## Source checkout (any OS, development)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=src
python -m twod2vr180.cli doctor
python -m twod2vr180.cli gui
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

## What gets installed where

| Location | Content |
|---|---|
| `%LOCALAPPDATA%\Programs\2D2VR180` | application (installer, per-user, no admin) |
| `%LOCALAPPDATA%\2D2VR180\runtimes\<id>` | isolated Python + PyTorch/CUDA wheels per backend family (created by `uv`) |
| `%LOCALAPPDATA%\2D2VR180\models\<id>` | model weights (downloaded after license acceptance) |
| `%LOCALAPPDATA%\2D2VR180\jobs\<job-id>` | inputs' frames, outputs, `run_report.json`, logs |

Set `TWOD2VR180_HOME` to relocate the data directory (e.g. to a larger drive).
