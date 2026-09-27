"""Model manager: manifest-driven, resumable, hash-verified downloads.

Rules (see .claude/skills/model-management): models are data, never source;
download from the original host; ``.part`` → verify → atomic rename; resume
interrupted downloads; show license before download; never bundle weights
whose license forbids redistribution.

SHA256 policy: when the manifest pins a SHA256 the file must match it. When
it does not (the hash could not be verified at manifest-authoring time), the
downloaded file's hash is recorded on first download ("trust on first use")
and every later load is verified against that record; the run report marks
such a model ``hash_status: tofu``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import app_paths, config_dir

ProgressFn = Callable[[int, int | None], None]


class ModelError(Exception):
    pass


class LicenseNotAccepted(ModelError):
    pass


@dataclass
class ModelFile:
    path: str               # relative path inside the model directory
    url: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass
class ModelEntry:
    id: str
    display_name: str
    source: str
    revision: str
    license: str
    license_url: str
    commercial_use: bool
    redistributable: bool
    required_vram_gb: float | None
    backends: list[str]
    files: list[ModelFile]
    requires_acceptance: bool = True
    gated: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def size_bytes(self) -> int | None:
        sizes = [f.size_bytes for f in self.files]
        return None if any(s is None for s in sizes) else sum(sizes)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        files = [ModelFile(**f) for f in d.get("files", [])]
        known = {k for k in cls.__dataclass_fields__ if k != "files"}
        base = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known and k != "files"}
        return cls(files=files, extra=extra, **base)


def load_manifest(path: Path | None = None) -> list[ModelEntry]:
    path = path or (config_dir() / "model-manifest.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ModelEntry.from_dict(m) for m in data["models"]]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ModelManager:
    def __init__(self, root: Path | None = None, manifest: list[ModelEntry] | None = None,
                 opener: Callable[..., object] | None = None):
        self.root = Path(root) if root else app_paths().models
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {m.id: m for m in (manifest if manifest is not None else load_manifest())}
        self._open = opener or urllib.request.urlopen

    # ------------------------------------------------------------ state
    def model_dir(self, model_id: str) -> Path:
        return self.root / model_id

    def _state_file(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "state.json"

    def state(self, model_id: str) -> dict:
        p = self._state_file(model_id)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _write_state(self, model_id: str, st: dict) -> None:
        p = self._state_file(model_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        os.replace(tmp, p)

    def accept_license(self, model_id: str) -> None:
        st = self.state(model_id)
        e = self.entries[model_id]
        st["license_accepted"] = {"license": e.license, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._write_state(model_id, st)

    def license_accepted(self, model_id: str) -> bool:
        e = self.entries[model_id]
        return (not e.requires_acceptance) or bool(self.state(model_id).get("license_accepted"))

    def is_installed(self, model_id: str) -> bool:
        e = self.entries.get(model_id)
        if e is None or not e.files:  # runtime-fetched models are not managed here
            return False
        st = self.state(model_id)
        files = st.get("files", {})
        for f in e.files:
            p = self.model_dir(model_id) / f.path
            rec = files.get(f.path)
            if not p.exists() or not rec:
                return False
            if rec.get("size") != p.stat().st_size:
                return False
        return True

    def status(self, model_id: str) -> dict:
        e = self.entries[model_id]
        st = self.state(model_id)
        hashes = {r.get("hash_status") for r in st.get("files", {}).values()}
        return {
            "id": e.id, "name": e.display_name, "installed": self.is_installed(model_id),
            "license": e.license, "license_url": e.license_url, "commercial_use": e.commercial_use,
            "redistributable": e.redistributable, "size_bytes": e.size_bytes, "source": e.source,
            "revision": e.revision, "license_accepted": self.license_accepted(model_id),
            "hash_status": "pinned" if hashes == {"pinned"} else ("tofu" if "tofu" in hashes else None),
            "gated": e.gated, "notes": e.notes,
        }

    def paths(self, model_id: str) -> dict[str, Path]:
        if not self.is_installed(model_id):
            raise ModelError(f"Model '{model_id}' is not installed. Download it from the Models tab "
                             f"or run: 2d2vr180 models download {model_id}")
        return {f.path: self.model_dir(model_id) / f.path for f in self.entries[model_id].files}

    # ------------------------------------------------------------ verify
    def verify(self, model_id: str, full_hash: bool = True) -> list[str]:
        e = self.entries[model_id]
        st = self.state(model_id)
        problems = []
        for f in e.files:
            p = self.model_dir(model_id) / f.path
            rec = st.get("files", {}).get(f.path)
            if not p.exists():
                problems.append(f"{f.path}: missing")
                continue
            if rec is None:
                problems.append(f"{f.path}: no download record")
                continue
            if rec.get("size") != p.stat().st_size:
                problems.append(f"{f.path}: size mismatch (corrupted or truncated)")
                continue
            if full_hash:
                want = f.sha256 or rec.get("sha256")
                if want and sha256_file(p) != want:
                    problems.append(f"{f.path}: SHA256 mismatch (corrupted)")
        return problems

    def delete(self, model_id: str) -> None:
        d = self.model_dir(model_id)
        if d.exists():
            shutil.rmtree(d)

    # ------------------------------------------------------------ download
    def download(self, model_id: str, progress: ProgressFn | None = None,
                 cancel: Callable[[], bool] | None = None, hf_token: str | None = None) -> dict[str, Path]:
        if model_id not in self.entries:
            raise ModelError(f"Unknown model '{model_id}'")
        e = self.entries[model_id]
        if not self.license_accepted(model_id):
            raise LicenseNotAccepted(f"License '{e.license}' for {e.display_name} must be accepted first "
                                     f"({e.license_url}).")
        d = self.model_dir(model_id)
        d.mkdir(parents=True, exist_ok=True)
        total = e.size_bytes
        done_before = 0
        st = self.state(model_id)
        st.setdefault("files", {})
        for f in e.files:
            target = d / f.path
            rec = st["files"].get(f.path)
            if target.exists() and rec and rec.get("size") == target.stat().st_size:
                done_before += target.stat().st_size
                continue
            self._check_disk(d, f.size_bytes)

            def _p(n: int, _base=done_before) -> None:
                if progress:
                    progress(_base + n, total)

            digest = self._download_file(f, target, _p, cancel, hf_token)
            if f.sha256:
                if digest != f.sha256.lower():
                    target.unlink(missing_ok=True)
                    raise ModelError(f"{f.path}: SHA256 mismatch (expected {f.sha256}, got {digest}). "
                                     "The download was discarded.")
                status = "pinned"
            else:
                status = "tofu"
            size = target.stat().st_size
            st["files"][f.path] = {"sha256": digest, "size": size, "hash_status": status, "url": f.url,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            st["revision"] = e.revision
            self._write_state(model_id, st)
            done_before += size
        return self.paths(model_id)

    def _check_disk(self, d: Path, need: int | None) -> None:
        if not need:
            return
        free = shutil.disk_usage(d).free
        if free < need * 1.05 + 256 * 1024 * 1024:
            raise ModelError(f"Not enough disk space in {d}: need {need / 2**30:.2f} GiB, "
                             f"have {free / 2**30:.2f} GiB free.")

    def _download_file(self, f: ModelFile, target: Path, progress: ProgressFn,
                       cancel: Callable[[], bool] | None, hf_token: str | None) -> str:
        part = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        have = 0
        if part.exists():
            with open(part, "rb") as pf:  # re-hash what we already have
                while True:
                    b = pf.read(1 << 22)
                    if not b:
                        break
                    h.update(b)
                    have += len(b)
        headers = {"User-Agent": "2D2VR180-model-manager"}
        if have:
            headers["Range"] = f"bytes={have}-"
        if hf_token and "huggingface.co" in f.url:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(f.url, headers=headers)
        attempts = 0
        while True:
            try:
                resp = self._open(req, timeout=60)
                break
            except urllib.error.HTTPError as ex:
                if ex.code == 416 and have:  # already complete
                    resp = None
                    break
                if ex.code in (401, 403):
                    raise ModelError(f"Access denied by {f.url} (HTTP {ex.code}). The model may be gated: "
                                     "accept its terms on the host and provide an access token.") from ex
                raise ModelError(f"Download failed: HTTP {ex.code} for {f.url}") from ex
            except (urllib.error.URLError, TimeoutError, OSError) as ex:
                attempts += 1
                if attempts >= 4:
                    raise ModelError(f"Network error downloading {f.url}: {ex}. Partial data kept; "
                                     "the next attempt resumes.") from ex
                time.sleep(2 ** attempts)
        if resp is not None:
            code = getattr(resp, "status", 200)
            mode = "ab"
            if have and code == 200:  # server ignored Range: restart
                h = hashlib.sha256()
                have = 0
                mode = "wb"
            with resp, open(part, mode) as out:
                while True:
                    if cancel and cancel():
                        raise ModelError("Download cancelled; partial data kept for resume.")
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    h.update(b)
                    have += len(b)
                    progress(have, None)
        if f.size_bytes and have != f.size_bytes:
            raise ModelError(f"{f.path}: size {have} != expected {f.size_bytes}; partial data kept.")
        os.replace(part, target)
        return h.hexdigest()
