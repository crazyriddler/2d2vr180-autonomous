# VR180 Skill

Never create stereoscopic depth by arbitrary pixel shifting when a 3D scene is available.

Preferred:
1. scene reconstruction;
2. define left/right virtual cameras;
3. render each eye;
4. handle disocclusion;
5. project to the requested VR180 format;
6. encode SBS or Top/Bottom;
7. validate orientation and dimensions.

Store:
- eye separation;
- FOV;
- projection;
- output resolution;
- frame rate;
- codec;
- pixel format.

The UI must state when the result is only a narrow-FOV stereo approximation rather than a true hemispherical VR180 reconstruction.
