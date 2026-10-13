"""Training-time augmentation of normalized (seq_len, F) clips.

All coordinates are already relative to a wrist or the shoulder midpoint, so scaling,
rotating and mirroring happen about the origin of each block. Flags are never changed except
by the flip, which swaps them together with the hands.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.features import FeatureLayout, interpolate
from src.landmarks import N_HAND


def mirror_permutation(layout: FeatureLayout, pose_pairs: list[list[int]], face_pairs: list[list[int]]) -> np.ndarray:
    """Column permutation that swaps every left-side value with its right-side counterpart."""
    pose_slot = {lm: i for i, lm in enumerate(layout.pose_indices)}
    face_slot = {lm: i for i, lm in enumerate(layout.face_indices)}
    pose_mirror = np.arange(len(layout.pose_indices))
    face_mirror = np.arange(len(layout.face_indices))
    for slots, mirror, pairs, name in ((pose_slot, pose_mirror, pose_pairs, "pose"),
                                       (face_slot, face_mirror, face_pairs, "face")):
        for a, b in pairs:
            if a not in slots or b not in slots:
                raise ValueError(f"{name} mirror pair {(a, b)} is not part of the feature subset")
            mirror[slots[a]], mirror[slots[b]] = slots[b], slots[a]

    point_map = np.concatenate([
        np.arange(N_HAND) + N_HAND,  # left hand slot i <- right hand slot i
        np.arange(N_HAND),
        2 * N_HAND + pose_mirror,
        2 * N_HAND + len(pose_mirror) + face_mirror,
    ])
    coord_cols = (point_map[:, None] * 3 + np.arange(3)).ravel()
    flags = layout.blocks["FLAGS"]
    return np.concatenate([coord_cols, [flags.start + 1, flags.start]])


def flip(clip: np.ndarray, layout: FeatureLayout, permutation: np.ndarray) -> np.ndarray:
    """Mirror a clip left-right: swap left/right parts and negate every x coordinate."""
    out = clip[:, permutation].copy()
    n_coords = layout.blocks["FLAGS"].start
    out[:, 0:n_coords:3] *= -1
    return out


def scale_rotate(clip: np.ndarray, layout: FeatureLayout, scale: float, angle_rad: float) -> np.ndarray:
    """Scale all coordinates and rotate x/y about each block's origin (a small camera roll)."""
    n_coords = layout.blocks["FLAGS"].start
    out = clip.copy()
    pts = out[:, :n_coords].reshape(len(clip), -1, 3)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y = pts[..., 0].copy(), pts[..., 1].copy()
    pts[..., 0] = scale * (c * x - s * y)
    pts[..., 1] = scale * (s * x + c * y)
    pts[..., 2] *= scale
    out[:, :n_coords] = pts.reshape(len(clip), -1)
    return out


def add_noise(clip: np.ndarray, layout: FeatureLayout, std: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian jitter on detected points only; missing parts stay exactly zero."""
    n_coords = layout.blocks["FLAGS"].start
    out = clip.copy()
    pts = out[:, :n_coords].reshape(len(clip), -1, 3)
    detected = np.any(pts != 0, axis=-1, keepdims=True)
    pts += rng.normal(0, std, pts.shape).astype(clip.dtype) * detected
    out[:, :n_coords] = pts.reshape(len(clip), -1)
    return out


def time_warp(clip: np.ndarray, gamma: float, start: float, end: float) -> np.ndarray:
    """Resample with a non-uniform clock: u**gamma speeds up one half of the sign and slows the other,
    and [start, end] crops the clip as an imprecise segmenter would."""
    t = len(clip)
    u = np.linspace(0, 1, t) ** gamma
    return interpolate(clip, (start + (end - start) * u) * (t - 1))


class Augmenter:
    """Draws random augmentation parameters per clip from the config."""

    def __init__(self, layout: FeatureLayout, cfg: dict[str, Any], seed: int):
        self.layout = layout
        self.aug = cfg["augment"]
        self.rng = np.random.default_rng(seed)
        feats = cfg["features"]
        self.permutation = mirror_permutation(layout, feats["pose_mirror_pairs"], feats["face_mirror_pairs"])

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        a, rng = self.aug, self.rng
        jitter = a["boundary_jitter"]
        out = time_warp(clip, rng.uniform(*a["time_warp_gamma"]), rng.uniform(0, jitter), 1 - rng.uniform(0, jitter))
        out = scale_rotate(out, self.layout, rng.uniform(*a["scale_range"]),
                           np.deg2rad(rng.uniform(-a["rotation_deg"], a["rotation_deg"])))
        out = add_noise(out, self.layout, a["noise_std"], rng)
        if a["flip"] and rng.random() < 0.5:
            out = flip(out, self.layout, self.permutation)
        return out.astype(clip.dtype)

    def batch(self, clips: np.ndarray) -> np.ndarray:
        return np.stack([self(c) for c in clips])
