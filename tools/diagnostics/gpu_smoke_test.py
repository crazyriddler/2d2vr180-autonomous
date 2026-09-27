"""Minimal GPU smoke test, run INSIDE a backend runtime:

    2d2vr180-cli runtimes install photo-cu128
    %LOCALAPPDATA%\\2D2VR180\\runtimes\\photo-cu128\\venv\\Scripts\\python.exe tools\\diagnostics\\gpu_smoke_test.py

Prints one JSON object: torch/CUDA versions, device, VRAM, a timed fp16 matmul,
peak allocation after a 12 GB-class allocation probe (skipped below 14 GB).
Exit code 0 = pass.
"""

import json
import sys
import time


def main() -> int:
    out = {"ok": False}
    try:
        import torch
    except ImportError as e:
        out["error"] = f"torch not importable: {e}"
        print(json.dumps(out, indent=2))
        return 2
    out.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        out["error"] = "CUDA not available (driver too old for this CUDA build, or no NVIDIA GPU)"
        print(json.dumps(out, indent=2))
        return 3
    p = torch.cuda.get_device_properties(0)
    out.update(device=p.name, capability=f"{p.major}.{p.minor}", vram_total_mib=p.total_memory // 2**20,
               target=("RTX 4080" in p.name))
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t
    out["fp16_tflops"] = round(50 * 2 * 4096 ** 3 / dt / 1e12, 2)
    out["matmul_finite"] = bool(torch.isfinite(c).all())
    del a, b, c
    if p.total_memory >= 14 * 2**30:
        x = torch.empty(int(12 * 2**30 // 2), dtype=torch.float16, device="cuda")
        x.fill_(1)
        del x
    torch.cuda.empty_cache()
    out["peak_alloc_mib"] = torch.cuda.max_memory_allocated() // 2**20
    out["ok"] = out["matmul_finite"]
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
