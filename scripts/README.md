# Scripts

All scripts are idempotent and log their actions. PowerShell scripts target
Windows (CI: `windows-latest`); the Python helpers run anywhere.

| Script | Purpose |
|---|---|
| `bootstrap.ps1` | build venv from `requirements-dev.txt`, fetch bundled tools |
| `fetch_build_tools.py` | download + SHA256-verify `uv.exe` and `ffmpeg.exe` (config/build-tools.lock.json) |
| `doctor.ps1` | hardware/driver diagnostics from source |
| `build.ps1` | PyInstaller build to `dist\2D2VR180` + frozen smoke test |
| `test.ps1` | pytest + frozen acceptance checks |
| `acceptance.py` | GPU-free acceptance checks against a built CLI (error paths, VR180 render) |
| `package.ps1` | portable ZIP, Inno Setup installer, checksums, release manifest |
| `install-test.ps1` | silent per-user install, run installed app without Python on PATH, acceptance, uninstall |
| `release.ps1` | refuses to release unless `docs/status/gates.json` mandatory gates pass |
| `release_manifest.py` | `checksums.sha256` + `release-manifest.json` |
| `resolve_upstreams.py` / `resolve-upstreams.ps1` | detect drift from `config/upstream-lock.json` |
| `resolve_models.py` | fill model revisions/SHA256/sizes from Hugging Face |
