# Release gate: refuses to proceed unless every mandatory acceptance gate in
# docs\status\gates.json is "pass". Publishing itself happens in CI as a DRAFT
# GitHub Release on a v* tag; a maintainer publishes it after reviewing the gates.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$gates = Get-Content docs\status\gates.json | ConvertFrom-Json
$failed = @()
foreach ($g in $gates.gates) { if ($g.mandatory -and $g.status -ne "pass") { $failed += "$($g.id): $($g.status)" } }
if ($failed.Count -gt 0) {
  Write-Host "Release blocked. Mandatory gates not passed:"; $failed | ForEach-Object { Write-Host "  $_" }
  exit 1
}
& $PSScriptRoot\bootstrap.ps1; & $PSScriptRoot\build.ps1; & $PSScriptRoot\test.ps1; & $PSScriptRoot\package.ps1
Write-Host "All gates pass. Push tag v<version> to create the draft GitHub Release."
