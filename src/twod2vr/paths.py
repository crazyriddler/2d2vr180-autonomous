"""Application directories and bundled-resource lookup.

All user data lives under %LOCALAPPDATA%/2D2VR on Windows (no admin rights
required) or ~/.local/share/2D2VR elsewhere. ``TWOD2VR_HOME`` overrides
the root, which the test-suite uses to stay hermetic. The 1.x folder (2D2VR180) with its tens of
gigabytes of models is renamed to the new one on first start, so nothing is downloaded again.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import APP_NAME, OLD_APP_NAME


def _default_root() -> Path:
    override = os.environ.get("TWOD2VR_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))
    return migrate_old_root(base / OLD_APP_NAME, base / APP_NAME)


def migrate_old_root(old: Path, new: Path) -> Path:
    """Take over the 1.x data folder (same drive: an instant rename). If it cannot be renamed (in use),
    keep using it where it is."""
    if new.exists() or not old.is_dir():
        return new
    try:
        old.rename(new)
        return new
    except OSError:
        return old


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def runtimes(self) -> Path:
        return self.root / "runtimes"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def settings_file(self) -> Path:
        return self.root / "settings.json"

    def ensure(self) -> "AppPaths":
        for p in (self.models, self.runtimes, self.jobs, self.cache, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def app_paths() -> AppPaths:
    return AppPaths(_default_root())


def resource_root() -> Path:
    """Directory holding shipped read-only resources (config/, workers/, bin/).

    In a PyInstaller build this is the bundle directory; in a source checkout it
    is the repository root.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen)
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return resource_root() / "config"


def workers_dir() -> Path:
    return resource_root() / "workers"


def bundled_bin_dir() -> Path:
    return resource_root() / "bin"
