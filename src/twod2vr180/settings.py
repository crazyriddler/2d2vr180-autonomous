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
    renderer: str = "auto"                       # auto | gpu | cpu
    fill_holes: bool = True
    vr180: bool = True
    layout_sbs: bool = True
    layout_tb: bool = True
    projection: str = "equirect180"              # equirect180 | flat
    eye_separation_mm: float = 64.0
    eye_resolution: int = 2048
    video_eye_resolution: int = 1280
    still_video_seconds: float = 5.0
    output_dir: str = ""                         # also copy results here ("" = keep in jobs folder)
    hf_token: str = ""                           # only for gated models; stored locally, never uploaded
    setup_completed: bool = False
    vr_help_seen: bool = False
    check_updates: bool = False                  # no network access unless the user enables it
    generative: str = "off"                      # off | arc | orbit | explore | spiral
    gen_assembly: str = "fusion"                 # fusion | train
    gen_engine: str = "auto"                     # auto | qwen | wan | seva
    video_mode: str = "auto"                     # auto | multiview | per_frame | best_frame
    combine_photos: bool = True                  # several photos dropped together → one multi-view scene
    ai_hole_fill: bool = True                    # LaMa inpainting of VR180 disocclusions
    export_sequence: bool = False                # fixed-camera video: one .ply per frame (4D)

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
        if self.renderer not in ("auto", "gpu", "cpu"):
            self.renderer = "auto"
        if self.generative not in ("off", "capture", "arc", "orbit", "explore", "spiral"):
            self.generative = "off"
        if self.gen_assembly not in ("fusion", "train"):
            self.gen_assembly = "fusion"
        if self.gen_engine not in ("auto", "qwen", "wan", "seva"):
            self.gen_engine = "auto"
        if self.video_mode not in ("auto", "multiview", "per_frame", "best_frame"):
            self.video_mode = "auto"
        if self.projection not in ("equirect180", "flat"):
            self.projection = "equirect180"
        self.eye_separation_mm = float(min(max(self.eye_separation_mm, 0.0), 200.0))
        self.eye_resolution = int(min(max(self.eye_resolution, 256), 4096))
        self.video_eye_resolution = int(min(max(self.video_eye_resolution, 256), 4096))
        self.still_video_seconds = float(min(max(self.still_video_seconds, 0.5), 120.0))

    def save(self, path: Path | None = None) -> None:
        path = Path(path or app_paths().settings_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def job_options(self, **overrides):
        from .jobs import JobOptions

        layouts = [x for x, on in (("sbs", self.layout_sbs), ("tb", self.layout_tb)) if on] or ["sbs"]
        o = JobOptions(mode=self.mode, vr180=self.vr180, layouts=layouts, projection=self.projection,
                       eye_resolution=self.eye_resolution, video_eye_resolution=self.video_eye_resolution,
                       eye_separation_m=self.eye_separation_mm / 1000.0,
                       still_video_seconds=self.still_video_seconds, license_profile=self.license_profile,
                       output_dir=self.output_dir or None, fill_holes=self.fill_holes, renderer=self.renderer,
                       allow_cpu=self.allow_cpu, generative=self.generative, gen_assembly=self.gen_assembly,
                       gen_engine=self.gen_engine,
                       video_mode=self.video_mode,
                       ai_hole_fill=self.ai_hole_fill, export_sequence=self.export_sequence)
        for k, v in overrides.items():
            setattr(o, k, v)
        return o
