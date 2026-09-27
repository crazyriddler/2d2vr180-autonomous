# PyInstaller spec: one folder containing 2D2VR180.exe (GUI) and 2d2vr180-cli.exe.
# Build:  pyinstaller packaging/2d2vr180.spec --noconfirm --distpath dist --workpath build/pyi
import os
from pathlib import Path

REPO = Path(SPECPATH).parent
datas = [
    (str(REPO / "config" / "model-manifest.json"), "config"),
    (str(REPO / "config" / "runtime-manifest.json"), "config"),
    (str(REPO / "config" / "upstream-lock.json"), "config"),
    (str(REPO / "config" / "build-tools.lock.json"), "config"),
    (str(REPO / "workers"), "workers"),
    (str(REPO / "LICENSE"), "."),
    (str(REPO / "THIRD_PARTY_NOTICES.md"), "."),
]
bin_dir = REPO / "build" / "bin"
binaries = [(str(p), "bin") for p in bin_dir.glob("*.exe")] if bin_dir.exists() else []
excludes = ["torch", "imageio_ffmpeg", "pytest", "tkinter", "matplotlib", "scipy", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore",
            "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf"]

common = dict(pathex=[str(REPO / "src")], binaries=binaries, datas=datas, hiddenimports=[],
              excludes=excludes, noarchive=False)
a_gui = Analysis([str(REPO / "packaging" / "launcher_gui.py")], **common)
a_cli = Analysis([str(REPO / "packaging" / "launcher_cli.py")], **common)

pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)
exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="2D2VR180", console=False,
              upx=False, version=None)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="2d2vr180-cli", console=True, upx=False)
coll = COLLECT(exe_gui, a_gui.binaries, a_gui.datas, exe_cli, a_cli.binaries, a_cli.datas,
               strip=False, upx=False, name="2D2VR180")
