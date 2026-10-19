"""User settings persisted as JSON in the app data directory."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import app_paths


@dataclass
class Settings:
    license_profile: str = "personal_research"   # personal_research | commercial
    mode: str = "auto"                           # auto | quality | fast
    allow_cpu: bool = False
    output_dir: str = ""                         # also copy results here ("" = keep in jobs folder)
    hf_token: str = ""                           # only for gated models; stored locally, never uploaded
    setup_completed: bool = False
    vr_help_seen: bool = False
    check_updates: bool = False                  # no network access unless the user enables it
    real3d_fill: bool = True                     # photos: AI-paint what the photo hides
    ipd_mm: float = 63.0                         # adults' eye distance used for the real-world scale
    photo_detail: bool = False                   # photos: maximum detail (always on in the Quality mode)
    video_seconds: float = 0.0                   # splat video: seconds used (0 = the whole video)
    video_fps: float = 0.0                       # splat video: frames per second (0 = the video's own, up to 24)
    video_start_s: float = 0.0                   # splat video: where to start

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or app_paths().settings_file
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in raw.items() if k in known})
        s.sanitize()
        return s

    def sanitize(self) -> None:
        if self.license_profile not in ("personal_research", "commercial"):
            self.license_profile = "personal_research"
        if self.mode not in ("auto", "quality", "fast"):
            self.mode = "auto"
        try:
            self.ipd_mm = float(min(max(float(self.ipd_mm), 45.0), 80.0))
        except (TypeError, ValueError):
            self.ipd_mm = 63.0
        self.video_seconds = float(min(max(self.video_seconds, 0.0), 3600.0))
        self.video_fps = float(min(max(self.video_fps, 0.0), 60.0))
        self.video_start_s = float(max(self.video_start_s, 0.0))

    def save(self, path: Path | None = None) -> None:
        path = Path(path or app_paths().settings_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def job_options(self, **overrides):
        from .jobs import JobOptions

        o = JobOptions(mode=self.mode, license_profile=self.license_profile, output_dir=self.output_dir or None,
                       allow_cpu=self.allow_cpu, real3d_fill=self.real3d_fill, ipd_mm=self.ipd_mm,
                       photo_detail=self.photo_detail,
                       video_seconds=self.video_seconds, video_fps=self.video_fps, video_start_s=self.video_start_s)
        for k, v in overrides.items():
            setattr(o, k, v)
        return o
