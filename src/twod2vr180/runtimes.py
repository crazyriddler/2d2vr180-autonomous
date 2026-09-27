"""Isolated ML runtimes (one per compatible dependency family).

The GUI process never imports torch. Each backend family gets its own
Python environment under ``<app-data>/runtimes/<id>``, created by a bundled
``uv`` binary from pinned wheels. Workers run as subprocesses of that
runtime's interpreter and talk JSON lines over stdout (see workers/_protocol.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import app_paths, bundled_bin_dir, config_dir

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class RuntimeInstallError(Exception):
    pass


@dataclass
class RuntimeSpec:
    id: str
    description: str
    python: str
    torch_index: str
    packages: list[str]
    archives_no_deps: list[dict] = field(default_factory=list)
    extra_index: str | None = None
    min_driver: str | None = None
    approx_size_gb: float | None = None
    status: str = "unverified"
    smoke_test: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def load_runtime_manifest(path: Path | None = None) -> dict[str, RuntimeSpec]:
    data = json.loads((path or config_dir() / "runtime-manifest.json").read_text(encoding="utf-8"))
    return {r["id"]: RuntimeSpec.from_dict(r) for r in data["runtimes"]}


def find_uv() -> str | None:
    exe = "uv.exe" if sys.platform == "win32" else "uv"
    cand = bundled_bin_dir() / exe
    if cand.exists():
        return str(cand)
    return shutil.which("uv")


class RuntimeManager:
    def __init__(self, root: Path | None = None, specs: dict[str, RuntimeSpec] | None = None):
        self.root = Path(root) if root else app_paths().runtimes
        self.specs = specs if specs is not None else load_runtime_manifest()

    def env_dir(self, rid: str) -> Path:
        return self.root / rid

    def python(self, rid: str) -> Path:
        d = self.env_dir(rid) / "venv"
        return d / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def _marker(self, rid: str) -> Path:
        return self.env_dir(rid) / "installed.json"

    def is_installed(self, rid: str) -> bool:
        m = self._marker(rid)
        if not (m.exists() and self.python(rid).exists()):
            return False
        try:
            rec = json.loads(m.read_text())
        except json.JSONDecodeError:
            return False
        return rec.get("spec_fingerprint") == self._fingerprint(rid)

    def _fingerprint(self, rid: str) -> str:
        import hashlib

        s = self.specs[rid]
        blob = json.dumps([s.python, s.torch_index, s.extra_index, s.packages, s.archives_no_deps], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def status(self, rid: str) -> dict:
        s = self.specs[rid]
        rec = {}
        if self._marker(rid).exists():
            try:
                rec = json.loads(self._marker(rid).read_text())
            except json.JSONDecodeError:
                rec = {}
        return {"id": rid, "description": s.description, "installed": self.is_installed(rid),
                "approx_size_gb": s.approx_size_gb, "manifest_status": s.status,
                "smoke_test": rec.get("smoke_test"), "installed_at": rec.get("at")}

    def worker_env(self, rid: str) -> dict[str, str]:
        """Environment for worker subprocesses: keep every cache inside the app
        directory and forbid implicit hub downloads unless a runtime needs them."""
        paths = app_paths()
        env = dict(os.environ)
        env["HF_HOME"] = str(paths.models / "hf-cache")
        env["TORCH_HOME"] = str(paths.models / "torch-cache")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        return env

    def install(self, rid: str, log: Callable[[str], None] = print,
                cancel: Callable[[], bool] | None = None) -> None:
        if rid not in self.specs:
            raise RuntimeInstallError(f"Unknown runtime '{rid}'")
        spec = self.specs[rid]
        uv = find_uv()
        if not uv:
            raise RuntimeInstallError("The bundled 'uv' tool is missing; reinstall 2D2VR180.")
        d = self.env_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(self.root / "_uv-cache")
        env["UV_PYTHON_INSTALL_DIR"] = str(self.root / "_python")
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
        env.pop("VIRTUAL_ENV", None)
        venv = d / "venv"

        def run(args: list[str]) -> None:
            log("$ " + " ".join(args))
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                    text=True, errors="replace", creationflags=_NO_WINDOW)
            assert proc.stdout is not None
            for line in proc.stdout:
                log(line.rstrip())
                if cancel and cancel():
                    proc.kill()
                    raise RuntimeInstallError("Runtime installation cancelled.")
            if proc.wait() != 0:
                raise RuntimeInstallError(f"Command failed ({proc.returncode}): {' '.join(args[:4])} ...")

        run([uv, "python", "install", spec.python])
        if not self.python(rid).exists():
            run([uv, "venv", "--python", spec.python, str(venv)])
        py = str(self.python(rid))
        idx = ["--index-url", spec.torch_index, "--extra-index-url", "https://pypi.org/simple",
               "--index-strategy", "unsafe-best-match"]
        if spec.extra_index:
            idx += ["--extra-index-url", spec.extra_index]
        run([uv, "pip", "install", "--python", py, *idx, *spec.packages])
        if spec.archives_no_deps:
            run([uv, "pip", "install", "--python", py, "--no-deps",
                 *[a["url"] for a in spec.archives_no_deps]])
        smoke = None
        if spec.smoke_test:
            r = subprocess.run([py, "-c", spec.smoke_test], capture_output=True, text=True,
                               env=self.worker_env(rid), creationflags=_NO_WINDOW)
            smoke = {"ok": r.returncode == 0, "stderr": r.stderr[-2000:]}
            log(f"smoke test: {'OK' if smoke['ok'] else 'FAILED'}\n{r.stderr[-2000:]}")
        self._marker(rid).write_text(json.dumps({
            "spec_fingerprint": self._fingerprint(rid),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "smoke_test": smoke,
        }, indent=2))
        if smoke and not smoke["ok"]:
            raise RuntimeInstallError(f"Runtime '{rid}' installed but its smoke test failed: "
                                f"{smoke['stderr'].strip().splitlines()[-1:] or ['?']}")

    def remove(self, rid: str) -> None:
        d = self.env_dir(rid)
        if d.exists():
            shutil.rmtree(d)

    def freeze(self, rid: str) -> list[str] | None:
        """Exact installed package versions, for run reports."""
        if not self.is_installed(rid):
            return None
        uv = find_uv()
        if not uv:
            return None
        r = subprocess.run([uv, "pip", "freeze", "--python", str(self.python(rid))], capture_output=True,
                           text=True, creationflags=_NO_WINDOW)
        return r.stdout.splitlines() if r.returncode == 0 else None
