"""Worker side of the 2D2VR180 worker protocol (stdlib only).

Workers run inside an isolated backend runtime. They receive one argument, the
path of a JSON request file, and report through JSON lines on stdout prefixed
with ``@@2D2VR180 `` so that library chatter on stdout is ignored safely:

    {"event": "progress", "value": 0.4, "message": "..."}
    {"event": "log", "message": "..."}
    {"event": "env", "torch": "...", "cuda": "...", "device": "...", "vram_total_mib": ...}
    {"event": "result", ...}          # exactly once on success
    {"event": "error", "code": "...", "message": "..."}   # exit code != 0

Error codes: oom, cuda_unavailable, model_missing, bad_input, upstream_error.
"""

import json
import sys
import traceback

PREFIX = "@@2D2VR180 "


def emit(event, **kw):
    kw["event"] = event
    sys.stdout.write(PREFIX + json.dumps(kw) + "\n")
    sys.stdout.flush()


def progress(value, message=""):
    emit("progress", value=float(value), message=message)


def log(message):
    emit("log", message=str(message))


def load_request():
    if len(sys.argv) != 2:
        emit("error", code="bad_input", message="usage: worker.py request.json")
        sys.exit(2)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        return json.load(f)


def torch_env(torch):
    info = {"torch": torch.__version__, "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device=p.name, vram_total_mib=int(p.total_memory // 2**20),
                    capability=f"{p.major}.{p.minor}")
    emit("env", **info)
    return info


def vram_peak_mib(torch):
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() // 2**20)
    return None


def run(main):
    """Run ``main(request)`` translating common failures into error events."""
    try:
        req = load_request()
        main(req)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - worker boundary
        name = type(e).__name__
        msg = str(e)
        code = "upstream_error"
        if "OutOfMemory" in name or "out of memory" in msg.lower():
            code = "oom"
        elif isinstance(e, FileNotFoundError):
            code = "model_missing"
        emit("error", code=code, message=f"{name}: {msg}", traceback=traceback.format_exc()[-4000:])
        sys.exit(1)
