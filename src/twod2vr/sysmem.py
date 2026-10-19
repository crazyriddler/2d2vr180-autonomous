"""System memory checks before heavy work (Windows first).

On Windows every allocation - in RAM and, through the WDDM driver, on the GPU too - needs "commit": room in
RAM plus the page file. When it runs out, programs do not get a clean error: CUDA reports "out of memory"
with gigabytes of VRAM free and memory-mapped model files crash the process (0xC0000005). Owner report on
rc52: both, at the start of a job. So before a job: our own worker processes left over from earlier jobs
(they are never needed between jobs, the queue runs one job at a time) are stopped, and how much memory
is left is logged, with a clear warning when it is too little.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
NEEDED_COMMIT_GIB = 24.0     # the AI image model alone keeps 15 GB in RAM


@dataclass
class MemoryStatus:
    ram_total_gib: float
    ram_free_gib: float
    commit_limit_gib: float       # RAM + page file
    commit_free_gib: float

    def summary(self) -> str:
        return (f"RAM {self.ram_free_gib:.1f} of {self.ram_total_gib:.1f} GB free; memory for programs (RAM + page "
                f"file) {self.commit_free_gib:.1f} of {self.commit_limit_gib:.1f} GB free")


def memory_status() -> MemoryStatus | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            g = 2 ** 30
            return MemoryStatus(st.ullTotalPhys / g, st.ullAvailPhys / g, st.ullTotalPageFile / g,
                                st.ullAvailPageFile / g)
        import os

        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page / 2 ** 30
        free = os.sysconf("SC_AVPHYS_PAGES") * page / 2 ** 30
        return MemoryStatus(total, free, total, free)
    except (AttributeError, OSError, ValueError):
        return None


def low_memory_warning(st: MemoryStatus | None, needed_gib: float = NEEDED_COMMIT_GIB) -> str | None:
    if st is None or st.commit_free_gib >= needed_gib:
        return None
    return (f"Windows has only {st.commit_free_gib:.1f} GB of memory left for programs (RAM + page file); "
            f"the AI models need about {needed_gib:.0f} GB. Close other programs (browsers, games, other AI "
            "tools), restart the PC if memory stays low, and let Windows manage the page file size "
            "(System > About > Advanced system settings > Performance > Virtual memory) on a drive with "
            "20 GB or more free.")


def stray_workers(runtimes_dir: Path) -> list[dict]:
    """Processes running an interpreter from the app's runtimes folder (Windows), as
    [{"pid", "mib", "exe"}]. Empty elsewhere or when the query fails."""
    if sys.platform != "win32":
        return []
    root = str(Path(runtimes_dir).resolve()).replace("'", "''")
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('"
          + root + "', [System.StringComparison]::OrdinalIgnoreCase) } | "
          "Select-Object ProcessId, WorkingSetSize, ExecutablePath | ConvertTo-Json -Compress")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, timeout=30, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_processes(r.stdout)


def parse_processes(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [{"pid": int(p["ProcessId"]), "mib": int((p.get("WorkingSetSize") or 0) // 2 ** 20),
             "exe": p.get("ExecutablePath") or ""} for p in data if p.get("ProcessId")]


def stop_stray_workers(runtimes_dir: Path, log=None) -> list[dict]:
    """Stop leftover worker processes of earlier jobs (only processes started from the app's own runtimes)."""
    procs = stray_workers(runtimes_dir)
    for p in procs:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p["pid"])], capture_output=True,
                       creationflags=_NO_WINDOW)
        if log:
            log(f"stopped a worker left over from an earlier job (PID {p['pid']}, {p['mib']:,} MB): {p['exe']}")
    return procs
