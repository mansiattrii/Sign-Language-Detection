import numpy as np
import pytest

from src.landmarks import FrameLandmarks
from src.segmenter import Segmenter, State, hand_motion

CFG = {"start_frames": 3, "preroll_frames": 2, "end_absent_frames": 4, "end_still_frames": 5, "max_frames": 50}

IDLE, MOVE, STILL = (False, False), (True, True), (True, False)


def run(events, cfg=CFG):
    """Feed (hand_present, moving) pairs; items are the frame numbers. Returns segments and states."""
    seg = Segmenter(cfg)
    segments, states = [], []
    for t, (present, moving) in enumerate(events):
        out = seg.update(t, present, moving)
        states.append(seg.state)
        if out is not None:
            segments.append(out)
    return segments, states


def test_sign_ends_when_hands_leave_and_includes_preroll():
    events = [IDLE] * 5 + [MOVE] * 10 + [IDLE] * 4
    segments, states = run(events)
    assert len(segments) == 1
    # Pre-roll keeps preroll_frames frames before the start_frames that triggered the start.
    assert segments[0].items == list(range(5 - 2, 15))
    assert segments[0].reason == "absent"
    assert states[7] is State.SIGNING and states[-1] is State.END


def test_short_hand_dropout_does_not_end_the_sign():
    events = [MOVE] * 6 + [IDLE] * 3 + [MOVE] * 6 + [IDLE] * 4
    segments, _ = run(events)
    assert len(segments) == 1
    assert segments[0].items == list(range(0, 15))


def test_sign_ends_when_hands_hold_still_and_drops_the_hold():
    events = [MOVE] * 8 + [STILL] * 5
    segments, _ = run(events)
    assert len(segments) == 1
    assert segments[0].reason == "still"
    assert segments[0].items == list(range(8))


def test_hands_resting_in_view_never_start_a_sign():
    segments, states = run([STILL] * 40)
    assert segments == [] and set(states) == {State.IDLE}


def test_start_needs_consecutive_moving_frames():
    events = [MOVE, MOVE, IDLE, MOVE, MOVE, IDLE] * 5
    segments, states = run(events)
    assert segments == [] and State.SIGNING not in states


def test_max_frames_forces_an_end():
    segments, _ = run([MOVE] * 60)
    assert segments[0].reason == "max_frames"
    assert len(segments[0].items) == CFG["max_frames"]


def test_two_signs_in_a_row_with_hands_still_between():
    events = [MOVE] * 6 + [STILL] * 5 + [STILL] * 3 + [MOVE] * 6 + [IDLE] * 4
    segments, _ = run(events)
    assert [s.reason for s in segments] == ["still", "absent"]
    assert segments[0].items == list(range(6))
    assert segments[1].items[-1] == 19


def _frame(hand=None, shoulder_width=0.5):
    pose = np.zeros((33, 4), np.float32)
    pose[11, 0], pose[12, 0] = shoulder_width, 0.0
    return FrameLandmarks(left_hand=hand, right_hand=None, pose=pose, face=None)


def test_hand_motion_in_shoulder_widths():
    hand = np.zeros((21, 3), np.float32)
    moved = hand.copy()
    moved[:, 0] += 0.1
    assert hand_motion(_frame(hand), _frame(moved)) == pytest.approx(0.2)


def test_hand_motion_appearing_and_missing_hands():
    hand = np.zeros((21, 3), np.float32)
    assert hand_motion(_frame(None), _frame(hand)) == float("inf")
    assert hand_motion(None, _frame(hand)) == float("inf")
    assert hand_motion(_frame(hand), _frame(None)) == 0.0
