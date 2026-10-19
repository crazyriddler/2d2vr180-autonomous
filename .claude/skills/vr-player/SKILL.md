# VR player skill (2D2VR 2.0)

The output is a 3D scene you walk into, never a flat stereo image: splats (.splat/.ply) for photos and
splat videos (.splv) for videos, played by `assets/webxr/player.*` (browser, WebXR) and the app's 3D viewer.

Rules:
1. The scene's real-world size comes from the reconstruction (eye distance, human model, measured depth);
   the player's size control (A/X smaller, B/Y bigger, stick click resets) only corrects what is off.
2. Playback speed is the video's: the sound is the clock; when a frame is not decoded yet, wait, never skip.
3. Memory is bounded: frames are streamed (HTTP range requests) and decoded a few seconds ahead.
4. Every .splv change keeps `workers/splv_codec.py` and the player's decoder in step (the format is
   documented at the top of splv_codec.py) and is covered by tests/integration/test_player.py.
5. Nothing leaves the machine: the player is served on 127.0.0.1 only.
