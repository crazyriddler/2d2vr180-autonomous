# Diagnostics

- `2d2vr180-cli doctor [--json]` — Windows version, NVIDIA driver, GPU model, total/free VRAM,
  CUDA driver API version, CPU, RAM, disk space, FFmpeg and uv availability. Exit code 3 when no
  NVIDIA GPU is visible.
- `gpu_smoke_test.py` — run with a backend runtime's interpreter to prove that PyTorch sees the GPU,
  a fp16 matmul is finite, and a 12 GB allocation succeeds on 16 GB cards.
- `2d2vr180-cli backends` — which backends can run here and exactly why not.
