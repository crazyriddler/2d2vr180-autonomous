# Installer smoke test: silent per-user install (no admin), run the INSTALLED
# app with Python/Git removed from PATH, run acceptance checks, uninstall.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$setup = Get-ChildItem release\2D2VR180-*-setup.exe | Select-Object -First 1
if (-not $setup) { throw "installer not found" }
$dir = Join-Path $env:LOCALAPPDATA "Programs\2D2VR180-installtest"
Start-Process $setup.FullName -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/CURRENTUSER", "/NORESTART", "/DIR=`"$dir`"" -Wait
$cli = Join-Path $dir "2d2vr180-cli.exe"
if (-not (Test-Path $cli)) { throw "installed CLI missing" }
if (-not (Test-Path (Join-Path $dir "2D2VR180.exe"))) { throw "installed GUI missing" }

# The installed app must not depend on a system Python, Git or CUDA toolkit.
$savedPath = $env:PATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
& $cli --version
if ($LASTEXITCODE -ne 0) { throw "installed CLI does not start without Python on PATH" }
& $cli doctor
Write-Host "doctor exit code: $LASTEXITCODE (3 = no NVIDIA GPU)"
$env:PATH = $savedPath

& .venv\Scripts\python.exe scripts\acceptance.py --app $cli
if ($LASTEXITCODE -ne 0) { throw "acceptance checks failed on the installed app" }

$unins = Join-Path $dir "unins000.exe"
Start-Process $unins -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait
Start-Sleep -Seconds 3
if (Test-Path $cli) { throw "uninstaller left the application behind" }
Write-Host "== install test passed (install, run without Python on PATH, acceptance, uninstall)"
