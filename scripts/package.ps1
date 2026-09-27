# Produce release\: portable ZIP, installer EXE, checksums, release manifest.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$version = & .venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import twod2vr180; print(twod2vr180.__version__)"
New-Item -ItemType Directory -Force release | Out-Null
Get-ChildItem release | Remove-Item -Force
Compress-Archive -Path dist\2D2VR180 -DestinationPath "release\2D2VR180-$version-portable-win64.zip" -CompressionLevel Optimal
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { choco install innosetup -y --no-progress; }
& $iscc /DAppVersion=$version /DSourceDir=..\dist\2D2VR180 /DOutputDir=..\release packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Copy-Item docs\RELEASE_NOTES.md "release\RELEASE_NOTES-$version.md"
& .venv\Scripts\python.exe scripts\release_manifest.py
if ($LASTEXITCODE -ne 0) { throw "release manifest failed" }
Get-ChildItem release
