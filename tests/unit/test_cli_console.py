import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_cli_survives_legacy_windows_code_page(tmp_path):
    """Regression (Windows CI): printing '→'/'°' on a cp1252 console crashed the CLI."""
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONPATH=str(REPO / "src"),
               TWOD2VR_HOME=str(tmp_path / "home"))
    r = subprocess.run([sys.executable, "-m", "twod2vr.cli", "backends"], capture_output=True, env=env)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert b"photo_locked" in r.stdout and b"splat_video" in r.stdout
