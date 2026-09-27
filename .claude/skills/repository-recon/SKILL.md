# Repository Recon Skill

When evaluating an upstream GitHub repository:

1. Clone it into a temporary workspace.
2. Inspect README, LICENSE, pyproject/package files, Dockerfiles, setup scripts and model download scripts.
3. Determine exact inference entry points.
4. Determine whether Windows is supported.
5. Determine CUDA/PyTorch versions.
6. Determine minimum VRAM.
7. Determine model weight license separately from code license.
8. Determine whether weights may be redistributed.
9. Run the smallest official example.
10. Record all findings in `config/upstream-lock.json`.

Never copy an upstream repository wholesale into the application unless its license and architecture make that appropriate.
