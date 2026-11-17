"""MediaPipe Holistic wrapper and the raw landmark containers shared by extraction and the live app.

mediapipe and cv2 are imported inside the functions that use them, so code that only needs the
containers (features, the API) does not pay for them.

Coordinates are stored with x and z multiplied by the image aspect ratio (width / height),
so all three axes are in units of image height. Without this, a 16:9 video and a 4:3 webcam
would stretch distances and angles differently.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np

N_HAND = 21
N_POSE = 33
N_FACE = 468

UPPER_BODY_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class FrameLandmarks:
    """Landmarks of a single frame; a part is None when Holistic did not detect it."""

    left_hand: np.ndarray | None  # (21, 3)
    right_hand: np.ndarray | None  # (21, 3)
    pose: np.ndarray | None  # (33, 4): x, y, z, visibility
    face: np.ndarray | None  # (468, 3)


@dataclass
class RawLandmarks:
    """Landmarks of a whole clip, with NaN rows for frames where a part was not detected."""

    left_hand: np.ndarray  # (T, 21, 3)
    right_hand: np.ndarray  # (T, 21, 3)
    pose: np.ndarray  # (T, 33, 4)
    face: np.ndarray  # (T, 468, 3)
    fps: float

    def __len__(self) -> int:
        return len(self.left_hand)

    @classmethod
    def from_frames(cls, frames: list[FrameLandmarks], fps: float) -> RawLandmarks:
        def stack(name: str, shape: tuple[int, int]) -> np.ndarray:
            out = np.full((len(frames), *shape), np.nan, dtype=np.float32)
            for t, frame in enumerate(frames):
                part = getattr(frame, name)
                if part is not None:
                    out[t] = part
            return out

        return cls(
            left_hand=stack("left_hand", (N_HAND, 3)),
            right_hand=stack("right_hand", (N_HAND, 3)),
            pose=stack("pose", (N_POSE, 4)),
            face=stack("face", (N_FACE, 3)),
            fps=fps,
        )

    def frames(self) -> list[FrameLandmarks]:
        """Split back into per-frame landmarks, as the live app receives them."""
        def part(arr: np.ndarray, t: int) -> np.ndarray | None:
            return None if np.isnan(arr[t]).all() else arr[t]

        return [FrameLandmarks(part(self.left_hand, t), part(self.right_hand, t), part(self.pose, t),
                               part(self.face, t)) for t in range(len(self))]

    def save(self, path: Path) -> None:
        """Write to an .npz file atomically, so an interrupted run never leaves a half-written cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, left_hand=self.left_hand, right_hand=self.right_hand,
                            pose=self.pose, face=self.face, fps=np.float32(self.fps))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> RawLandmarks:
        with np.load(path) as d:
            return cls(d["left_hand"], d["right_hand"], d["pose"], d["face"], float(d["fps"]))


def _to_array(landmark_list: Any, aspect: float, with_visibility: bool = False) -> np.ndarray | None:
    if landmark_list is None:
        return None
    rows = [
        (p.x * aspect, p.y, p.z * aspect, p.visibility) if with_visibility else (p.x * aspect, p.y, p.z * aspect)
        for p in landmark_list.landmark
    ]
    return np.asarray(rows, dtype=np.float32)


@lru_cache(maxsize=1)
def hand_connections() -> list[tuple[int, int]]:
    import mediapipe as mp

    return sorted(mp.solutions.hands.HAND_CONNECTIONS)


class HolisticTracker:
    """Runs MediaPipe Holistic frame by frame.

    Holistic tracks across frames in video mode, so use a fresh tracker for every clip.
    Frames must not be mirrored: Holistic labels hands from the signer's point of view,
    and a mirrored frame would swap the left- and right-hand blocks.
    """

    def __init__(self, holistic_cfg: dict[str, Any]):
        import mediapipe as mp

        self.max_frame_width = holistic_cfg.get("max_frame_width")
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=holistic_cfg["model_complexity"],
            min_detection_confidence=holistic_cfg["min_detection_confidence"],
            min_tracking_confidence=holistic_cfg["min_tracking_confidence"],
            refine_face_landmarks=holistic_cfg["refine_face_landmarks"],
        )

    def process(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        import cv2

        h, w = frame_bgr.shape[:2]
        if self.max_frame_width and w > self.max_frame_width:
            scale = self.max_frame_width / w
            frame_bgr = cv2.resize(frame_bgr, (self.max_frame_width, round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._holistic.process(rgb)
        aspect = w / h
        return FrameLandmarks(
            left_hand=_to_array(res.left_hand_landmarks, aspect),
            right_hand=_to_array(res.right_hand_landmarks, aspect),
            pose=_to_array(res.pose_landmarks, aspect, with_visibility=True),
            face=_to_array(res.face_landmarks, aspect),
        )

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> HolisticTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_video_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield BGR frames of a video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def video_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else 30.0


def extract_video(path: Path, holistic_cfg: dict[str, Any]) -> RawLandmarks:
    """Run Holistic over every frame of a video."""
    with HolisticTracker(holistic_cfg) as tracker:
        frames = [tracker.process(frame) for frame in iter_video_frames(path)]
    return RawLandmarks.from_frames(frames, video_fps(path))


def draw_landmarks(image: np.ndarray, frame: FrameLandmarks, face_indices: list[int],
                   pose_indices: list[int]) -> np.ndarray:
    """Draw the hands, the used pose points and the face subset onto a BGR image in place."""
    import cv2

    h = image.shape[0]

    # x was multiplied by width/height, so pixel x is simply x * height.
    def px(p: np.ndarray) -> tuple[int, int]:
        return int(p[0] * h), int(p[1] * h)

    for hand, color in ((frame.left_hand, (255, 128, 0)), (frame.right_hand, (0, 128, 255))):
        if hand is None:
            continue
        for a, b in hand_connections():
            cv2.line(image, px(hand[a]), px(hand[b]), color, 2)
        for p in hand:
            cv2.circle(image, px(p), 3, (255, 255, 255), -1)

    if frame.pose is not None:
        for a, b in UPPER_BODY_CONNECTIONS:
            cv2.line(image, px(frame.pose[a]), px(frame.pose[b]), (0, 220, 0), 2)
        for i in pose_indices:
            cv2.circle(image, px(frame.pose[i]), 5, (0, 220, 0), -1)

    if frame.face is not None:
        for i in face_indices:
            cv2.circle(image, px(frame.face[i]), 2, (0, 0, 255), -1)
    return image
