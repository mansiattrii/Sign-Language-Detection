"""Feature vectors built from raw Holistic landmarks.

This module is the single place where landmarks become model input. Extraction, training
and the live app all call build_sequence(), so the same clip always yields the same array.

Per-frame vector layout:
    LEFT_HAND (63) | RIGHT_HAND (63) | POSE (21) | FACE (3 * n_face) | FLAGS (2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.landmarks import N_HAND, RawLandmarks

HAND_DIM = N_HAND * 3
WRIST, MIDDLE_MCP = 0, 9
LEFT_SHOULDER, RIGHT_SHOULDER, NOSE = 11, 12, 0


@dataclass(frozen=True)
class FeatureLayout:
    """Positions of each landmark group inside the per-frame vector."""

    pose_indices: tuple[int, ...]
    face_indices: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> FeatureLayout:
        feats = cfg["features"]
        face = tuple(i for region in feats["face_regions"].values() for i in region)
        return cls(tuple(feats["pose_indices"]), face)

    @property
    def blocks(self) -> dict[str, slice]:
        pose_start = 2 * HAND_DIM
        face_start = pose_start + 3 * len(self.pose_indices)
        flags_start = face_start + 3 * len(self.face_indices)
        return {
            "LEFT_HAND": slice(0, HAND_DIM),
            "RIGHT_HAND": slice(HAND_DIM, 2 * HAND_DIM),
            "HANDS": slice(0, 2 * HAND_DIM),
            "POSE": slice(pose_start, face_start),
            "FACE": slice(face_start, flags_start),
            "FLAGS": slice(flags_start, flags_start + 2),
        }

    @property
    def size(self) -> int:
        return self.blocks["FLAGS"].stop

    def columns(self, *groups: str) -> np.ndarray:
        """Column indices for the named groups, e.g. columns("HANDS", "FLAGS")."""
        all_cols = np.arange(self.size)
        return np.concatenate([all_cols[self.blocks[g]] for g in groups])


def normalize_hand(hand: np.ndarray, eps: float) -> np.ndarray:
    """Translate each frame's hand to its wrist and scale by the wrist-to-middle-MCP length.

    hand: (T, 21, 3) with NaN rows for missing frames. Missing frames become zeros.
    The length is 3D: when a hand points at the camera its 2D length collapses (down to
    4% of normal on INCLUDE), which blew normalized fingertips up to 30 units.
    """
    present = ~np.isnan(hand[:, WRIST, 0])
    size = np.linalg.norm(hand[:, MIDDLE_MCP] - hand[:, WRIST], axis=-1)
    out = (hand - hand[:, WRIST:WRIST + 1]) / np.maximum(size, eps)[:, None, None]
    out[~present] = 0.0
    return out


def shoulder_valid(pose: np.ndarray, min_visibility: float, eps: float) -> np.ndarray:
    """Frames where both shoulders are detected, visible enough and not on top of each other."""
    ls, rs = pose[:, LEFT_SHOULDER], pose[:, RIGHT_SHOULDER]
    with np.errstate(invalid="ignore"):
        visible = (ls[:, 3] >= min_visibility) & (rs[:, 3] >= min_visibility)
        width = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=-1)
        return visible & (width > eps)


def body_reference(pose: np.ndarray, min_visibility: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame origin (shoulder midpoint) and scale (shoulder width) for pose and face points.

    Frames without usable shoulders reuse the last valid reference; frames before the
    first valid one use the first valid one. A clip with no usable shoulders at all
    falls back to the nose as origin with unit scale.
    """
    t = len(pose)
    valid = shoulder_valid(pose, min_visibility, eps)
    if not valid.any():
        return pose[:, NOSE, :3].copy(), np.ones(t, dtype=pose.dtype)

    ls, rs = pose[:, LEFT_SHOULDER, :3], pose[:, RIGHT_SHOULDER, :3]
    center = (ls + rs) / 2
    width = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=-1)

    source = np.where(valid, np.arange(t), -1)
    source = np.maximum.accumulate(source)
    source[source < 0] = np.flatnonzero(valid)[0]
    return center[source], width[source]


def frame_features(raw: RawLandmarks, layout: FeatureLayout, cfg: dict[str, Any]) -> np.ndarray:
    """Normalized per-frame feature vectors, shape (T, layout.size), float32."""
    feats = cfg["features"]
    eps = feats["eps"]
    t = len(raw)

    left = normalize_hand(raw.left_hand, eps).reshape(t, -1)
    right = normalize_hand(raw.right_hand, eps).reshape(t, -1)

    center, scale = body_reference(raw.pose, feats["shoulder_min_visibility"], eps)
    center, scale = center[:, None, :], np.maximum(scale, eps)[:, None, None]
    pose = (raw.pose[:, list(layout.pose_indices), :3] - center) / scale
    face = (raw.face[:, list(layout.face_indices)] - center) / scale

    flags = np.stack([~np.isnan(raw.left_hand[:, WRIST, 0]), ~np.isnan(raw.right_hand[:, WRIST, 0])], axis=1)

    out = np.concatenate([left, right, pose.reshape(t, -1), face.reshape(t, -1), flags], axis=1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def trim_to_hands(features: np.ndarray, layout: FeatureLayout) -> np.ndarray:
    """Drop leading and trailing frames in which neither hand is present."""
    present = np.flatnonzero(features[:, layout.blocks["FLAGS"]].any(axis=1))
    if len(present) == 0:
        return features[:0]
    return features[present[0]:present[-1] + 1]


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    """Linearly interpolate a (T, F) sequence to (length, F) along the time axis."""
    t = len(sequence)
    if t == 0:
        raise ValueError("Cannot resample an empty sequence")
    if t == 1:
        return np.repeat(sequence, length, axis=0)
    pos = np.linspace(0, t - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo)[:, None]
    return ((1 - w) * sequence[lo] + w * sequence[hi]).astype(sequence.dtype)


def count_hand_frames(features: np.ndarray, layout: FeatureLayout) -> int:
    return int(features[:, layout.blocks["FLAGS"]].any(axis=1).sum())


def build_sequence(raw: RawLandmarks, layout: FeatureLayout, cfg: dict[str, Any]) -> np.ndarray | None:
    """Full pipeline for one clip: normalize, trim, resample to seq_len.

    Returns None when the clip has too few frames with a hand to be a usable sign.
    """
    temporal = cfg["temporal"]
    feats = trim_to_hands(frame_features(raw, layout, cfg), layout)
    if count_hand_frames(feats, layout) < temporal["min_hand_frames"]:
        return None
    return resample(feats, temporal["seq_len"])
