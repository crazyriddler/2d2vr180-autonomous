# Hardware / driver / tool diagnostics from a source checkout.
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "src"
& .venv\Scripts\python.exe -m twod2vr180.cli doctor @args
exit $LASTEXITCODE
