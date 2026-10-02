"""CI check (no GPU, no 25 GB download): load_pipeline() assembles the Wan 2.2 pipeline the way
the worker uses it - pre-encoded prompt, no text encoder, model offload hooks, VAE eviction.

Usage: <gen-cu128 python> scripts/wan_assembly_check.py <repo>/workers
"""

import os
import sys

sys.path.insert(0, sys.argv[1])
os.environ["VIDEOX_ATTENTION_TYPE"] = "SDPA"
import torch  # noqa: E402
import wan_worker as w  # noqa: E402

w.stub_triton()
import videox_fun.models as vm  # noqa: E402


class _Dummy(torch.nn.Module):
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    device = property(lambda self: self.p.device)
    dtype = property(lambda self: self.p.dtype)
    config = property(lambda self: {})


class _Loader:
    @staticmethod
    def from_pretrained(*a, **k):
        return _Dummy()


vm.Wan2_2Transformer3DModel = _Loader
vm.AutoencoderKLWan3_8 = _Loader
pos, neg = torch.zeros(7, 8, dtype=torch.bfloat16), torch.ones(9, 8, dtype=torch.bfloat16)
pipe = w.load_pipeline("unused", "model_cpu_offload", torch, (pos, neg))
p, n = pipe.encode_prompt("x", None, True, device=torch.device("cpu"))
assert pipe.text_encoder.dtype == torch.bfloat16
assert p[0].shape == (7, 8) and n[0].shape == (9, 8)
assert "prepare_and_evict" in pipe.prepare_mask_latents.__name__
print("WAN_PIPELINE_OK")
