"""Resolve model metadata (exact revision, file sizes, LFS SHA256, licence, gating) from
Hugging Face and GitHub release assets. Runs in CI (the dev container has no HF access);
the JSON it prints is copied into config/model-manifest.json.

    python scripts/resolve_models.py > resolved.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request

HF_REPOS = [
    "Qwen/Qwen-Image-Edit-2511", "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    "lightx2v/Qwen-Image-Edit-2511-Lightning", "unsloth/Qwen-Image-Edit-2511-GGUF",
    "QuantStack/Qwen-Image-Edit-2511-GGUF",
]
HF_SEARCH = ["Qwen-Image-Edit-2511 gguf", "Qwen-Image-Edit-2511-Multiple-Angles"]
GGUF_HEADERS = [  # tensor names of GGUF files (read from the first MiBs, not downloaded)
    "unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q5_K_M.gguf",
]
HF_TEXT = [  # small files whose content is needed (configs, model cards)
    "Qwen/Qwen-Image-Edit-2511/transformer/config.json", "Qwen/Qwen-Image-Edit-2511/model_index.json",
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/README.md",
]
URLS: list[str] = []  # non-HF downloads: hashed by streaming


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "2D2VR180-resolver"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def gguf_tensor_names(url: str, nbytes: int = 16 << 20) -> dict:
    """Parse a GGUF header (metadata + tensor infos) from the start of the file."""
    import struct

    req = urllib.request.Request(url, headers={"User-Agent": "2D2VR180-resolver", "Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        b = r.read()
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, b, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]

    def rstr():
        n = rd("Q")
        nonlocal pos
        s = b[pos:pos + n].decode("utf-8", "replace")
        pos += n
        return s

    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [rval(et) for _ in range(n)][:8]
        return rd(sizes[t])

    assert b[:4] == b"GGUF"
    pos = 4
    version, n_tensors, n_kv = rd("I"), rd("Q"), rd("Q")
    meta = {}
    for _ in range(n_kv):
        k = rstr()
        meta[k] = rval(rd("I"))
    names = []
    for _ in range(n_tensors):
        name = rstr()
        nd = rd("I")
        dims = [rd("Q") for _ in range(nd)]
        ttype, _off = rd("I"), rd("Q")
        names.append([name, dims, ttype])
    return {"version": version, "tensors": n_tensors, "meta": {k: v for k, v in meta.items() if "token" not in k},
            "first": names[:40], "last": names[-15:]}


def main() -> int:
    out = {"hf": {}, "urls": {}}
    for repo in HF_REPOS:
        try:
            info = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
            files = {}
            for s in info.get("siblings", []):
                lfs = s.get("lfs") or {}
                files[s["rfilename"]] = {"size": lfs.get("size") or s.get("size"), "sha256": lfs.get("sha256")}
            out["hf"][repo] = {"revision": info.get("sha"), "gated": info.get("gated"),
                               "license": (info.get("cardData") or {}).get("license"),
                               "license_name": (info.get("cardData") or {}).get("license_name"),
                               "license_link": (info.get("cardData") or {}).get("license_link"),
                               "files": files}
        except Exception as e:  # noqa: BLE001
            out["hf"][repo] = {"error": f"{type(e).__name__}: {e}"}
    out["search"] = {}
    for q in HF_SEARCH:
        try:
            res = get("https://huggingface.co/api/models?limit=30&search=" + urllib.parse.quote(q))
            out["search"][q] = [{"id": m.get("id"), "downloads": m.get("downloads"), "likes": m.get("likes")}
                                for m in res]
        except Exception as e:  # noqa: BLE001
            out["search"][q] = {"error": f"{type(e).__name__}: {e}"}
    out["text"] = {}
    for spec in HF_TEXT:
        org, name, path = spec.split("/", 2)
        try:
            req = urllib.request.Request(f"https://huggingface.co/{org}/{name}/resolve/main/{path}",
                                         headers={"User-Agent": "2D2VR180-resolver"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out["text"][spec] = r.read(60000).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            out["text"][spec] = f"error: {type(e).__name__}: {e}"
    out["gguf"] = {}
    for spec in GGUF_HEADERS:
        org, name, path = spec.split("/", 2)
        try:
            out["gguf"][spec] = gguf_tensor_names(f"https://huggingface.co/{org}/{name}/resolve/main/{path}")
        except Exception as e:  # noqa: BLE001
            out["gguf"][spec] = {"error": f"{type(e).__name__}: {e}"}
    for url in URLS:
        try:
            h = hashlib.sha256()
            n = 0
            req = urllib.request.Request(url, headers={"User-Agent": "2D2VR180-resolver"})
            with urllib.request.urlopen(req, timeout=300) as r:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    h.update(b)
                    n += len(b)
            out["urls"][url] = {"size": n, "sha256": h.hexdigest()}
        except Exception as e:  # noqa: BLE001
            out["urls"][url] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
