"""The whole subject of a photo in 3D - back included - with TRELLIS.2-4B (runtime 'subject3d-cu128').

TRELLIS.2 (Microsoft, MIT) turns one image into a textured mesh; only its shape stages run here (sparse
structure -> shape latents, 512 -> 1024 cascade -> mesh), the texture models are not needed or downloaded. The Windows build of its CUDA extensions
comes from the TRELLIS.2-stableprojectorz fork (MIT, pinned in config/upstream-lock.json). Here it only
provides geometry: the multi-view worker aligns the mesh to the photo (workers/subject_align.py) and uses
its depth in the AI views as a shape prior that is consistent all the way round.

Request:
  {"image": "...", "output_dir": "...", "model_dir": "<TRELLIS.2-4B>", "image_large_dir": "<TRELLIS-image-large>",
   "dinov3_zip": "...", "rmbg_zip": "...", "pipeline_type": "1024_cascade", "seed": 42}
Result: outputs.mesh = subject.npz (vertices (N,3), faces (M,3), colors (N,3) in 0..1, TRELLIS frame);
        outputs.mask = the photo's subject mask (PNG, photo size) from the same background removal.
"""

import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _protocol import emit, log, progress, run, torch_env, vram_peak_mib  # noqa: E402


def unpacked(zip_path):
    """Directory with the zip's content, extracted next to it on first use (model files are verified
    by the model manager as zips; extraction is local)."""
    dest = zip_path[:-4] if zip_path.lower().endswith(".zip") else zip_path + "_unpacked"
    marker = os.path.join(dest, ".unpacked")
    if not os.path.exists(marker):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        open(marker, "w").close()
    # zips may hold a top-level folder
    entries = [e for e in os.listdir(dest) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest, entries[0])) and \
            not os.path.exists(os.path.join(dest, "config.json")):
        return os.path.join(dest, entries[0])
    return dest


def local_pipeline_dir(req, work):
    """A pipeline.json whose every model points to local files (no Hub access at inference)."""
    with open(os.path.join(req["model_dir"], "pipeline.json")) as f:
        cfg = json.load(f)
    args = cfg["args"]
    for k, v in list(args["models"].items()):
        if k.startswith("tex_"):                       # geometry only
            del args["models"][k]
            continue
        if v.startswith("microsoft/TRELLIS-image-large/"):
            args["models"][k] = os.path.join(req["image_large_dir"], v[len("microsoft/TRELLIS-image-large/"):])
        elif not os.path.isabs(v):
            args["models"][k] = os.path.join(req["model_dir"], v)
    args["image_cond_model"]["args"]["model_name"] = unpacked(req["dinov3_zip"])
    args["rembg_model"]["args"]["model_name"] = unpacked(req["rmbg_zip"])
    d = os.path.join(work, "pipeline")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pipeline.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    return d


def subject_mask(pipe, image, torch):
    """Background removal at the photo's size (the same model TRELLIS.2 uses), as a bool array."""
    import numpy as np

    rgba = pipe.rembg_model(image.convert("RGB"))
    a = np.asarray(rgba)[:, :, 3]
    if a.shape != (image.height, image.width):
        from PIL import Image

        a = np.asarray(Image.fromarray(a).resize((image.width, image.height), Image.BILINEAR))
    return a > 127


def shape_only(pipe, image, ptype, torch):
    """Trellis2ImageTo3DPipeline.run() without the texture stages: the subject's mesh."""
    img = pipe.preprocess_image(image)
    if pipe.low_vram:
        pipe.image_cond_model.to(pipe.device)
    cond_512 = pipe._cast_cond(pipe.get_cond([img], 512))
    cond_1024 = pipe._cast_cond(pipe.get_cond([img], 1024)) if ptype != "512" else None
    if pipe.low_vram:
        pipe.image_cond_model.cpu()
        torch.cuda.empty_cache()
    progress(0.25, "sparse structure")
    coords = pipe.sample_sparse_structure(cond_512, 32, 1, {})
    progress(0.45, "shape")
    if ptype == "512":
        slat = pipe.sample_shape_slat(cond_512, pipe.models["shape_slat_flow_model_512"], coords, {})
        res = 512
    else:
        slat, res = pipe.sample_shape_slat_cascade(cond_512, cond_1024, pipe.models["shape_slat_flow_model_512"],
                                                   pipe.models["shape_slat_flow_model_1024"], 512, 1024, coords,
                                                   {}, 49152)
    del cond_512, cond_1024
    slat._spatial_cache = {}
    torch.cuda.empty_cache()
    progress(0.8, "mesh")
    dec = pipe.models["shape_slat_decoder"]          # as Trellis2ImageTo3DPipeline.decode_latent prepares it
    if dec.dtype != torch.float16:
        dec.convert_to_fp16()
        dec.dtype = torch.float16
    if pipe.low_vram:
        dec.low_vram = True
        dec.from_latent.to(pipe.device)
        dec.output_layer.to(pipe.device)
    meshes, _ = pipe.decode_shape_slat(slat, res)
    if pipe.low_vram:
        dec.cpu()
    torch.cuda.empty_cache()
    return meshes[0]


def main(req):
    os.environ.setdefault("ATTN_BACKEND", "xformers")          # no flash-attn wheel for Windows
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    env = torch_env(torch)
    if not env["cuda_available"]:
        emit("error", code="cuda_unavailable", message="TRELLIS.2 needs an NVIDIA GPU (CUDA).")
        sys.exit(1)
    out = req["output_dir"]
    os.makedirs(out, exist_ok=True)
    progress(0.02, "preparing TRELLIS.2 (local models)")
    pdir = local_pipeline_dir(req, out)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    progress(0.05, "loading TRELLIS.2-4B")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(pdir)
    pipe.low_vram = True
    pipe.cuda()
    with Image.open(req["image"]) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    progress(0.1, "subject mask")
    with torch.no_grad():
        pipe.rembg_model.to("cuda")
        mask = subject_mask(pipe, image, torch)
        pipe.rembg_model.cpu()
    torch.cuda.empty_cache()
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out, "subject_mask.png"))
    if mask.sum() < 500:
        emit("error", code="no_subject", message="No clear subject found in the photo for TRELLIS.2.")
        sys.exit(1)
    ptype = req.get("pipeline_type", "1024_cascade")
    progress(0.15, f"building the whole subject in 3D (TRELLIS.2 shape, {ptype})")
    torch.manual_seed(int(req.get("seed", 42)))
    with torch.no_grad():
        mesh = shape_only(pipe, image, ptype, torch)
        try:
            mesh.simplify(int(req.get("max_faces", 2_000_000)))
        except Exception as e:  # noqa: BLE001 - a dense mesh is fine, only slower to sample
            log(f"mesh simplification skipped: {e}")
    colors = None
    v = mesh.vertices.float().cpu().numpy()
    f = mesh.faces.long().cpu().numpy()
    np.savez_compressed(os.path.join(out, "subject.npz"), vertices=v.astype(np.float32), faces=f.astype(np.int32),
                        colors=(colors if colors is not None else np.full_like(v, 0.5)).astype(np.float32))
    log(f"TRELLIS.2 subject: {len(v):,} vertices, {len(f):,} faces")
    emit("result", outputs={"mesh": os.path.join(out, "subject.npz"), "mask": os.path.join(out, "subject_mask.png")},
         vertices=int(len(v)), faces=int(len(f)), pipeline_type=ptype, vram_peak_mib=vram_peak_mib(torch))


if __name__ == "__main__":
    run(main)
