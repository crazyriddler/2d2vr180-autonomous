# Release Agent Prompt

Take the current repository and produce the best release that is actually reproducible.

Do not add new research features during release hardening.

Checklist:
- resolve versions;
- verify licenses;
- build;
- test;
- package;
- checksum;
- clean-install;
- smoke-test;
- generate release notes;
- generate release manifest.

If a backend fails, disable it rather than hiding the failure.
