# GitHub Storage Strategy

GitHub currently blocks ordinary repository files above 100 MiB and recommends keeping repositories much smaller than the hard on-disk limits. Releases allow binary assets under 2 GiB each and up to 1000 assets per release.

Therefore:

## Never put these in Git

- `.safetensors`
- `.pth`
- `.pt`
- `.ckpt`
- `.onnx` model files
- CUDA toolkits
- PyTorch wheels
- FFmpeg archives
- packaged application builds
- datasets
- sample videos
- generated scenes
- caches
- virtual environments

## Keep in Git

- source code;
- scripts;
- manifests;
- hashes;
- configuration;
- documentation;
- small test fixtures;
- lock files;
- GitHub Actions workflows.

## Release assets

Use GitHub Releases for:
- Windows installer;
- portable ZIP;
- checksums;
- release manifest.

Each release asset must remain below 2 GiB.

If a runtime bundle exceeds 2 GiB:
- split by component;
- or use an external object store/model host;
- or make the application download individual runtimes/models after installation.

Do not split files merely to evade limits if it creates an unusable distribution.

## Model distribution

Default strategy:

```text
GitHub repository
  ├── model-manifest.json
  ├── downloader
  └── checksums

Original model host
  ├── model A
  ├── model B
  └── model C

User PC
  └── %LOCALAPPDATA%/2D2VR180/models/
```

The downloader must display:
- source;
- license;
- size;
- expected hash;
- whether redistribution is permitted.

## Git LFS

Use Git LFS only for files that truly belong to source control and are legally redistributable. Do not use LFS as a dumping ground for model weights.

## Build artifacts

CI artifacts are for development/testing. Published end-user binaries go to GitHub Releases.

## Reproducibility

Every release contains a machine-readable manifest with:
- source commit;
- upstream commit;
- model revision;
- hashes;
- build timestamp;
- compiler/runtime versions.
