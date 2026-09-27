"""Apple SHARP single-image 3DGS worker (runs in runtime 'photo-cu128').

Request: {"checkpoint": ".../sharp_2572gikvuh.pt", "input_dir": "...", "output_dir": "..."}
SHARP writes <stem>.ply (INRIA 3DGS layout + SHARP metadata elements, OpenCV
convention, sRGB colours) per input image.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env  # noqa: E402


def main(req):
    import torch

    env = torch_env(torch)
    if not env["cuda_available"] and not req.get("allow_cpu", False):
        emit("error", code="cuda_unavailable", message="CUDA is not available to PyTorch in this runtime.")
        sys.exit(1)
    if not os.path.exists(req["checkpoint"]):
        emit("error", code="model_missing", message=f"SHARP checkpoint not found: {req['checkpoint']}")
        sys.exit(1)
    progress(0.05, "running SHARP")
    cmd = [sys.executable, "-c", "from sharp.cli import main_cli; main_cli()", "predict",
           "-i", req["input_dir"], "-o", req["output_dir"], "-c", req["checkpoint"]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        log(line)
    if proc.wait() != 0:
        msg = "\n".join(tail[-10:])
        code = "oom" if "out of memory" in msg.lower() else "upstream_error"
        emit("error", code=code, message="SHARP failed:\n" + msg)
        sys.exit(1)
    plys = sorted(f for f in os.listdir(req["output_dir"]) if f.endswith(".ply"))
    if not plys:
        emit("error", code="upstream_error", message="SHARP finished without writing a .ply")
        sys.exit(1)
    emit("result", outputs=[os.path.join(req["output_dir"], p) for p in plys], vram_peak_mib=None)


if __name__ == "__main__":
    run(main)
