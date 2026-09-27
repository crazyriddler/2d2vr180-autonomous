Set-Location (Join-Path $PSScriptRoot "..")
& .venv\Scripts\python.exe scripts\resolve_upstreams.py @args
exit $LASTEXITCODE
