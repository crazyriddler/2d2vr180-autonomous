# Backend Integration Skill

Every backend must implement the internal adapter contract.

Required methods:
- prepare
- can_run
- estimate
- run
- export
- cleanup

Requirements:
- no hard-coded user paths;
- no global model downloads during module import;
- progress events;
- cancellation;
- structured errors;
- deterministic working directory;
- cleanup;
- output validation.

Do not modify upstream code unless necessary. Prefer a thin adapter or a small maintained patch.
