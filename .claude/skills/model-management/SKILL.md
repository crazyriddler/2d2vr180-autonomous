# Model Management Skill

Models are data, not source.

Rules:
- never commit model weights;
- download from original/authorized hosts;
- record revision and SHA256;
- download to `.part`;
- verify hash;
- atomically rename;
- resume interrupted downloads;
- support delete/re-download;
- show license before download;
- do not bundle non-redistributable weights.

Model storage:
`%LOCALAPPDATA%/2D2VR180/models/<model-id>/`

A release is invalid if its model manifest points to a dead URL or an unverified license.
