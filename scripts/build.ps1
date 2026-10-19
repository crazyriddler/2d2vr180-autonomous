# Build the PyInstaller one-folder application into dist\2D2VR. Idempotent (clean build).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path dist\2D2VR) { Remove-Item -Recurse -Force dist\2D2VR }
& .venv\Scripts\python.exe -m PyInstaller packaging\2d2vr.spec --noconfirm --distpath dist --workpath build\pyi
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
# Frozen smoke test: the packaged CLI must start and find its bundled config/tools.
& dist\2D2VR\2d2vr-cli.exe --version
if ($LASTEXITCODE -ne 0) { throw "frozen CLI does not start" }
& dist\2D2VR\2d2vr-cli.exe models list
if ($LASTEXITCODE -ne 0) { throw "frozen CLI cannot read its model manifest" }
if (-not (Test-Path dist\2D2VR\_internal\bin\ffmpeg.exe)) { throw "ffmpeg.exe not bundled" }
if (-not (Test-Path dist\2D2VR\_internal\bin\uv.exe)) { throw "uv.exe not bundled" }
Write-Host "== build done"
