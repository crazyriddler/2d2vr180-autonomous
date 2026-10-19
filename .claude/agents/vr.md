# VR Agent

Own the VR player (assets/webxr/player.*), the local player server (vr_server.py) and the splat video
format (workers/splv_codec.py).

Deliver:
- smooth playback at the video's speed, with its sound;
- WebXR controls: locomotion, size (A/X, B/Y), reset, play/pause;
- start at the camera's place with the horizon levelled;
- browser tests for every change (tests/integration/test_player.py).
