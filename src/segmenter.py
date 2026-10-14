"""Cuts a continuous stream of frames into single signs, without needing a webcam to test.

States:
    IDLE     waiting; a sign starts once a hand has been visible *and moving* for start_frames
             frames in a row. Requiring motion means hands resting in view do not trigger it.
    SIGNING  collecting frames; the sign ends when the hands leave the view for end_absent_frames,
             stay still for end_still_frames, or the sign reaches max_frames.
    END      reported for the single frame on which a segment is returned; the next frame is
             handled as IDLE again.

The segmenter does not care what a frame is: it stores whatever item it is given (landmarks,
or landmarks plus the camera image) and returns those items for each finished sign.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.landmarks import FrameLandmarks

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


class State(Enum):
    IDLE = "IDLE"
    SIGNING = "SIGNING"
    END = "END"


@dataclass
class Segment:
    """Frames of one sign, from the pre-roll up to the last frame with a moving hand."""

    items: list[Any]
    reason: str  # "absent", "still" or "max_frames"


def hand_motion(prev: FrameLandmarks | None, cur: FrameLandmarks, eps: float = 1e-6) -> float:
    """Mean movement of the 21 hand points since the previous frame, in shoulder widths.

    All points, not just the wrist, so that a handshape change with a fixed wrist counts as
    motion. Dividing by shoulder width makes the threshold independent of camera distance (it
    falls back to image-height units if the shoulders are missing). A hand that has just
    appeared counts as moving; no hand at all counts as still.
    """
    moves = []
    for side in ("left_hand", "right_hand"):
        now = getattr(cur, side)
        if now is None:
            continue
        before = getattr(prev, side) if prev is not None else None
        if before is None:
            return float("inf")
        moves.append(float(np.linalg.norm(now[:, :2] - before[:, :2], axis=1).mean()))
    if not moves:
        return 0.0
    scale = 1.0
    if cur.pose is not None:
        scale = max(float(np.linalg.norm(cur.pose[LEFT_SHOULDER, :2] - cur.pose[RIGHT_SHOULDER, :2])), eps)
    return max(moves) / scale


class Segmenter:
    def __init__(self, seg_cfg: dict[str, Any]):
        self.start_frames = seg_cfg["start_frames"]
        self.preroll_frames = seg_cfg["preroll_frames"]
        self.end_absent_frames = seg_cfg["end_absent_frames"]
        self.end_still_frames = seg_cfg["end_still_frames"]
        self.max_frames = seg_cfg["max_frames"]
        self.reset()

    def reset(self) -> None:
        self.state = State.IDLE
        self._preroll: deque[Any] = deque(maxlen=self.preroll_frames + self.start_frames)
        self._start_run = 0
        self._buffer: list[Any] = []
        self._active_end = 0
        self._absent = 0
        self._still = 0

    def update(self, item: Any, hand_present: bool, moving: bool) -> Segment | None:
        """Feed one frame. Returns the finished sign on the frame where it ends, else None."""
        if self.state is State.END:
            self.reset()
        if self.state is State.IDLE:
            return self._idle(item, hand_present and moving)
        return self._signing(item, hand_present, moving)

    def _idle(self, item: Any, active: bool) -> None:
        self._preroll.append(item)
        self._start_run = self._start_run + 1 if active else 0
        if self._start_run >= self.start_frames:
            self.state = State.SIGNING
            self._buffer = list(self._preroll)
            self._active_end = len(self._buffer)
        return None

    def _signing(self, item: Any, hand_present: bool, moving: bool) -> Segment | None:
        self._buffer.append(item)
        self._absent = 0 if hand_present else self._absent + 1
        self._still = self._still + 1 if hand_present and not moving else 0
        if hand_present and moving:
            self._active_end = len(self._buffer)

        reason = None
        if self._absent >= self.end_absent_frames:
            reason = "absent"
        elif self._still >= self.end_still_frames:
            reason = "still"
        elif len(self._buffer) >= self.max_frames:
            reason = "max_frames"
        if reason is None:
            return None
        self.state = State.END
        # Trailing frames without hands or with the hands held still are not part of the sign.
        return Segment(self._buffer[:self._active_end], reason)
