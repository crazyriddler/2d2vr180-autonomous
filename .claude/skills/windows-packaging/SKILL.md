# Windows Packaging Skill

Target:
- Windows 11
- RTX 4080 16 GB

Build:
1. clean build directory;
2. build backend workers;
3. package GUI;
4. package native libraries;
5. run smoke tests;
6. build installer;
7. install in clean VM;
8. run end-to-end tests;
9. generate checksums;
10. generate release manifest.

Avoid requiring admin rights.

Do not ship a compiler toolchain unless a runtime backend actually needs it at execution time.
