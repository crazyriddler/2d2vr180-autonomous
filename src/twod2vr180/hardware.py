"""Hardware probe.

The application process never imports torch/CUDA. GPU facts come from
``nvidia-smi`` (installed with every NVIDIA driver on Windows), which is enough
to decide what can run before a heavy runtime is started. Backend workers
report the CUDA runtime they actually loaded in their own result events.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

TARGET_GPU = "NVIDIA GeForce RTX 4080"
TARGET_VRAM_MIB = 16 * 1024


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int
    driver: str
    compute_capability: str | None = None
    cuda_driver_version: str | None = None

    @property
    def is_target(self) -> bool:
        return "RTX 4080" in self.name


@dataclass
class HardwareReport:
    os: str
    os_version: str
    python: str
    cpu: str
    cpu_count: int
    ram_total_mib: int | None
    disk_free_mib: int | None
    disk_path: str
    gpus: list[GpuInfo] = field(default_factory=list)
    nvidia_smi: str | None = None
    ffmpeg: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def best_gpu(self) -> GpuInfo | None:
        return max(self.gpus, key=lambda g: g.total_mib, default=None)

    @property
    def cuda_available(self) -> bool:
        return self.best_gpu is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        best = self.best_gpu
        d["best_gpu_index"] = best.index if best else None
        d["target_gpu_detected"] = bool(best and best.is_target)
        return d


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, creationflags=flags
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        for base in (os.environ.get("SystemRoot", r"C:\Windows") + r"\System32",
                     r"C:\Program Files\NVIDIA Corporation\NVSMI"):
            cand = Path(base) / "nvidia-smi.exe"
            if cand.exists():
                return str(cand)
    return None


def parse_nvidia_smi_csv(text: str) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total,memory.free,
    driver_version,compute_cap --format=csv,noheader,nounits`` output."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                total_mib=int(float(parts[2])),
                free_mib=int(float(parts[3])),
                driver=parts[4],
                compute_capability=parts[5] if len(parts) > 5 and parts[5] not in ("", "[N/A]") else None,
            ))
        except ValueError:
            continue
    return gpus


def probe_gpus() -> tuple[list[GpuInfo], str | None, list[str]]:
    warnings: list[str] = []
    smi = find_nvidia_smi()
    if not smi:
        return [], None, ["nvidia-smi not found: no NVIDIA driver detected; GPU backends disabled."]
    out = _run([smi, "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    if out is None:  # older drivers lack compute_cap
        out = _run([smi, "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits"])
    if out is None:
        return [], smi, ["nvidia-smi failed to run; the NVIDIA driver may be broken."]
    gpus = parse_nvidia_smi_csv(out)
    header = _run([smi]) or ""
    cuda_ver = None
    for token in header.split("|"):
        if "CUDA Version" in token:
            cuda_ver = token.split("CUDA Version:")[-1].strip()
    for g in gpus:
        g.cuda_driver_version = cuda_ver
    if gpus and not any(g.is_target for g in gpus):
        warnings.append(f"Primary target is {TARGET_GPU}; detected {', '.join(g.name for g in gpus)}. "
                        "Defaults will scale to available VRAM.")
    return gpus, smi, warnings


def _ram_total_mib() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys // (1024 * 1024))
        except Exception:
            return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return None


def _cpu_name() -> str:
    name = platform.processor()
    if not name and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return name or platform.machine()


def probe(disk_path: Path | None = None) -> HardwareReport:
    from .media import find_ffmpeg  # local import: media imports nothing from here
    from .paths import app_paths

    disk_path = disk_path or app_paths().root
    probe_path = disk_path
    while not probe_path.exists() and probe_path.parent != probe_path:
        probe_path = probe_path.parent
    try:
        disk_free = shutil.disk_usage(probe_path).free // (1024 * 1024)
    except OSError:
        disk_free = None
    gpus, smi, warnings = probe_gpus()
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        warnings.append("FFmpeg not found: video input and VR180 video encoding disabled.")
    return HardwareReport(
        os=platform.system(),
        os_version=platform.version() if sys.platform == "win32" else platform.release(),
        python=sys.version.split()[0],
        cpu=_cpu_name(),
        cpu_count=os.cpu_count() or 1,
        ram_total_mib=_ram_total_mib(),
        disk_free_mib=disk_free,
        disk_path=str(disk_path),
        gpus=gpus,
        nvidia_smi=smi,
        ffmpeg=ffmpeg,
        warnings=warnings,
    )


def format_report(r: HardwareReport) -> str:
    lines = [
        f"OS:        {r.os} {r.os_version}",
        f"CPU:       {r.cpu} ({r.cpu_count} threads)",
        f"RAM:       {r.ram_total_mib} MiB" if r.ram_total_mib else "RAM:       unknown",
        f"Disk free: {r.disk_free_mib} MiB at {r.disk_path}",
        f"FFmpeg:    {r.ffmpeg or 'NOT FOUND'}",
    ]
    if not r.gpus:
        lines.append("GPU:       none detected (NVIDIA driver/nvidia-smi missing)")
    for g in r.gpus:
        tag = "  <- target" if g.is_target else ""
        lines.append(f"GPU {g.index}:     {g.name}, {g.total_mib} MiB total / {g.free_mib} MiB free, "
                     f"driver {g.driver}, CUDA driver API {g.cuda_driver_version or '?'}, "
                     f"sm {g.compute_capability or '?'}{tag}")
    for w in r.warnings:
        lines.append(f"WARNING:   {w}")
    return "\n".join(lines)
