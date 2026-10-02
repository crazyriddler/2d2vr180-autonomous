import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import wan_worker as w  # noqa: E402


def test_arc_has_side_and_vertical_shots_starting_at_the_photo():
    plan = w.shots("arc", 33)
    assert len(plan) == 4 and w.shot_labels("arc") == ["right", "left", "from above", "from below"]
    for shot in plan:
        assert len(shot) == 33 and np.allclose(shot[0], np.eye(4))


def test_vertical_shots_look_at_the_subject_from_above_and_below():
    above, below = w.shots("arc", 33)[2][-1], w.shots("arc", 33)[3][-1]
    target = np.array([0.0, 0.0, 2.0])
    for m in (above, below):
        fwd = m[:3, :3] @ np.array([0, 0, 1.0])
        to_target = target - m[:3, 3]
        assert np.dot(fwd, to_target / np.linalg.norm(to_target)) > 0.999   # still aimed at the subject
    assert above[1, 3] < 0 and below[1, 3] > 0      # OpenCV: y points down


def test_key_frames_spread_over_the_shot():
    k = w.key_frames(49)
    assert len(k) == 4 and k[-1] == 48 and 0 not in k
    e = w.ease_curve(49)
    assert np.allclose(e[k], [0.25, 0.5, 0.75, 1.0], atol=0.05)


def test_16gb_cards_start_with_fp8_weights():
    a16 = w.memory_attempts(16376, 704)
    assert a16[0] == ("model_cpu_offload_and_qfloat8", 704) and a16[-1][0] == "sequential_cpu_offload"
    assert w.memory_attempts(24576, 704)[0] == ("model_cpu_offload", 704)
