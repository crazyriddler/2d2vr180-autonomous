# Unit + integration tests (GPU-free) and a frozen-app smoke test when a build exists.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:QT_QPA_PLATFORM = "offscreen"
& .venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }
if (Test-Path dist\2D2VR180\2d2vr180-cli.exe) {
  $env:TWOD2VR180_HOME = Join-Path $env:RUNNER_TEMP "2d2vr180-home"
  & dist\2D2VR180\2d2vr180-cli.exe doctor
  Write-Host "doctor exit code: $LASTEXITCODE (3 = no NVIDIA GPU, expected on hosted runners)"
  & .venv\Scripts\python.exe scripts\acceptance.py --app dist\2D2VR180\2d2vr180-cli.exe
  if ($LASTEXITCODE -ne 0) { throw "acceptance checks failed" }
}
Write-Host "== tests done"
