# Create the build virtualenv and fetch bundled tools. Idempotent.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Write-Host "== bootstrap: python $(python --version)"
if (-not (Test-Path .venv)) { python -m venv .venv }
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .venv\Scripts\python.exe scripts\fetch_build_tools.py
if ($LASTEXITCODE -ne 0) { throw "fetch_build_tools failed" }
Write-Host "== bootstrap done"
