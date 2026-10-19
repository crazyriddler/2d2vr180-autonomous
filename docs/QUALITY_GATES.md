# Quality Gates

A backend is not promoted to production merely because it runs.

## Gate A — installation

- clean Windows test VM;
- no Python;
- no Git;
- no CUDA toolkit;
- no Conda;
- no Node;
- no Visual Studio.

## Gate B — hardware

- RTX 4080 detected;
- 16 GB VRAM detected;
- driver visible;
- no unexplained OOM.

## Gate C — correctness

For each supported input:
- output exists;
- output opens;
- source-view render is visually faithful;
- no catastrophic geometry explosion;
- no crash on cancellation.

## Gate D — novel views

Render at:
- source camera;
- small horizontal offset;
- ±15°;
- ±30°;
- ±45° when supported.

Record failures.

## Gate E — splat video playback

Verify:
- the browser player and the app viewer play a .splv in order, at the video's speed, with its sound;
- seeking and looping;
- frames are waited for, never skipped, when decoding is late;
- the VR size control (A/X, B/Y, stick click) and play/pause (trigger) on a PC-connected headset;
- memory stays bounded while playing a long splat video.

## Gate F — generated content honesty

A report must distinguish:
- source-observed;
- reconstructed;
- inferred;
- generatively completed.

## Gate G — licensing

No release is allowed if:
- a model cannot legally be redistributed in the planned form;
- an attribution/license file is missing;
- a non-commercial checkpoint is accidentally bundled into a release intended for commercial distribution.

## Gate H — reproducibility

Build the same release twice or at minimum reproduce the same source/runtime manifest.

## Gate I — recovery

Test:
- interrupted model download;
- insufficient disk space;
- corrupted model;
- missing model;
- unsupported GPU;
- invalid media;
- cancelled job;
- application restart after failed job.
