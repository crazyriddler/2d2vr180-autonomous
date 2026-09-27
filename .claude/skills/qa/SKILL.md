# QA Skill

Never accept a green exit code as sufficient.

Validate:
- files exist;
- output dimensions;
- non-zero file size;
- parseability;
- source-view similarity;
- camera trajectory;
- VR180 eye ordering;
- no NaN/Inf in 3D data;
- VRAM peak;
- runtime;
- logs.

Keep small deterministic fixtures in `tests/fixtures/` only if licensing permits.
Large media belongs outside Git.
