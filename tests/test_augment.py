import numpy as np
import pytest

from src.augment import Augmenter, add_noise, flip, mirror_permutation, scale_rotate, time_warp
from src.config import load_config
from src.features import FeatureLayout

CFG = load_config("config.yaml")
LAYOUT = FeatureLayout.from_config(CFG)
PERM = mirror_permutation(LAYOUT, CFG["features"]["pose_mirror_pairs"], CFG["features"]["face_mirror_pairs"])
B = LAYOUT.blocks
RNG = np.random.default_rng(0)


def random_clip(t: int = 30) -> np.ndarray:
    clip = RNG.normal(size=(t, LAYOUT.size)).astype(np.float32)
    clip[:, B["FLAGS"]] = 1.0
    return clip


def points(clip: np.ndarray, block: str) -> np.ndarray:
    return clip[:, B[block]].reshape(len(clip), -1, 3)


def test_double_flip_returns_the_original():
    clip = random_clip()
    assert np.array_equal(flip(flip(clip, LAYOUT, PERM), LAYOUT, PERM), clip)


def test_mirror_permutation_is_an_involution():
    assert sorted(PERM) == list(range(LAYOUT.size))
    assert np.array_equal(PERM[PERM], np.arange(LAYOUT.size))


def test_flip_moves_left_hand_to_right_and_mirrors_x():
    clip = random_clip(4)
    clip[:, B["RIGHT_HAND"]] = 0
    clip[:, B["FLAGS"]] = [1, 0]
    out = flip(clip, LAYOUT, PERM)
    left, right = points(clip, "LEFT_HAND"), points(out, "RIGHT_HAND")
    assert np.all(out[:, B["LEFT_HAND"]] == 0)
    assert np.all(out[:, B["FLAGS"]] == [0, 1])
    assert np.allclose(right[..., 0], -left[..., 0])
    assert np.allclose(right[..., 1:], left[..., 1:])


def test_flip_swaps_shoulders_and_keeps_a_symmetric_pose():
    clip = random_clip(2)
    pose = points(clip, "POSE")
    slot = {lm: i for i, lm in enumerate(LAYOUT.pose_indices)}
    pose[:, slot[0]] = [0.0, -0.6, 0.1]
    pose[:, slot[11]] = [0.5, 0.0, 0.0]
    pose[:, slot[12]] = [-0.5, 0.0, 0.0]
    clip[:, B["POSE"]] = pose.reshape(2, -1)
    out = points(flip(clip, LAYOUT, PERM), "POSE")
    for lm in (0, 11, 12):
        assert np.allclose(out[:, slot[lm]], pose[:, slot[lm]])
    assert np.allclose(out[:, slot[13], 0], -pose[:, slot[14], 0])


def test_flip_swaps_face_mirror_pairs_and_keeps_midline_points():
    clip = random_clip(2)
    face, out = points(clip, "FACE"), points(flip(clip, LAYOUT, PERM), "FACE")
    slot = {lm: i for i, lm in enumerate(LAYOUT.face_indices)}
    for a, b in CFG["features"]["face_mirror_pairs"]:
        assert np.allclose(out[:, slot[a], 0], -face[:, slot[b], 0])
        assert np.allclose(out[:, slot[a], 1:], face[:, slot[b], 1:])
    for lm in (0, 1, 17):
        assert np.allclose(out[:, slot[lm], 0], -face[:, slot[lm], 0])


def test_mirror_pair_outside_subset_is_rejected():
    with pytest.raises(ValueError):
        mirror_permutation(LAYOUT, [[11, 12]], [[70, 999]])


def test_rotation_preserves_lengths_and_flags():
    clip = random_clip()
    out = scale_rotate(clip, LAYOUT, 1.0, np.deg2rad(10))
    before = np.linalg.norm(points(clip, "HANDS")[..., :2], axis=-1)
    after = np.linalg.norm(points(out, "HANDS")[..., :2], axis=-1)
    assert np.allclose(before, after, atol=1e-5)
    assert np.array_equal(out[:, B["FLAGS"]], clip[:, B["FLAGS"]])


def test_noise_leaves_missing_hand_at_zero():
    clip = random_clip()
    clip[:, B["LEFT_HAND"]] = 0
    out = add_noise(clip, LAYOUT, 0.05, RNG)
    assert np.all(out[:, B["LEFT_HAND"]] == 0)
    assert not np.allclose(out[:, B["RIGHT_HAND"]], clip[:, B["RIGHT_HAND"]])


def test_time_warp_identity_and_crop():
    clip = random_clip()
    assert np.allclose(time_warp(clip, 1.0, 0.0, 1.0), clip)
    cropped = time_warp(clip, 1.0, 0.1, 0.9)
    assert cropped.shape == clip.shape
    assert not np.allclose(cropped[0], clip[0])


def test_augmenter_keeps_shape_dtype_and_is_seeded():
    clips = np.stack([random_clip() for _ in range(3)])
    a = Augmenter(LAYOUT, CFG, seed=1).batch(clips)
    b = Augmenter(LAYOUT, CFG, seed=1).batch(clips)
    assert a.shape == clips.shape and a.dtype == clips.dtype
    assert np.array_equal(a, b)
