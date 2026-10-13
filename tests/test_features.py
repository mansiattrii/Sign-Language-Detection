import numpy as np
import pytest

from src.config import load_config
from src.features import (FeatureLayout, body_reference, build_sequence, frame_features, normalize_hand, resample,
                          trim_to_hands)
from src.landmarks import N_FACE, N_HAND, N_POSE, RawLandmarks

CFG = load_config("config.yaml")
LAYOUT = FeatureLayout.from_config(CFG)
EPS = CFG["features"]["eps"]
RNG = np.random.default_rng(0)


def make_raw(t: int, left: bool = True, right: bool = True) -> RawLandmarks:
    pose = RNG.uniform(0.2, 0.8, (t, N_POSE, 4)).astype(np.float32)
    pose[:, :, 3] = 1.0
    pose[:, 11, :3] = [0.6, 0.5, 0.0]
    pose[:, 12, :3] = [0.4, 0.5, 0.0]
    raw = RawLandmarks(
        left_hand=RNG.uniform(0.3, 0.7, (t, N_HAND, 3)).astype(np.float32),
        right_hand=RNG.uniform(0.3, 0.7, (t, N_HAND, 3)).astype(np.float32),
        pose=pose,
        face=RNG.uniform(0.3, 0.7, (t, N_FACE, 3)).astype(np.float32),
        fps=25.0,
    )
    if not left:
        raw.left_hand[:] = np.nan
    if not right:
        raw.right_hand[:] = np.nan
    return raw


def test_layout_sizes_match_spec():
    blocks = LAYOUT.blocks
    assert len(LAYOUT.face_indices) == 53
    assert blocks["HANDS"] == slice(0, 126)
    assert blocks["POSE"] == slice(126, 147)
    assert blocks["FACE"] == slice(147, 306)
    assert blocks["FLAGS"] == slice(306, 308)
    assert LAYOUT.size == 308
    assert list(LAYOUT.columns("HANDS", "FLAGS")) == list(range(126)) + [306, 307]


def test_hand_is_wrist_centred_and_unit_size():
    hand = normalize_hand(make_raw(5).left_hand, EPS)
    assert np.allclose(hand[:, 0], 0.0)
    assert np.allclose(np.linalg.norm(hand[:, 9], axis=-1), 1.0)


def test_hand_scale_survives_pointing_at_the_camera():
    hand = make_raw(1).left_hand
    hand[0, 0] = [0.5, 0.5, 0.0]
    hand[0, 9] = [0.5, 0.501, -0.1]
    assert np.abs(normalize_hand(hand, EPS)).max() < 20


def test_hand_normalization_is_translation_and_scale_invariant():
    hand = make_raw(3).left_hand
    moved = hand * 2.5 + np.array([0.1, -0.2, 0.05], dtype=np.float32)
    assert np.allclose(normalize_hand(hand, EPS), normalize_hand(moved, EPS), atol=1e-5)


def test_degenerate_hand_does_not_divide_by_zero():
    hand = np.full((2, N_HAND, 3), 0.5, dtype=np.float32)
    out = normalize_hand(hand, EPS)
    assert np.all(np.isfinite(out))


def test_missing_hand_gives_zero_block_and_zero_flag():
    feats = frame_features(make_raw(4, left=False), LAYOUT, CFG)
    blocks = LAYOUT.blocks
    assert np.all(feats[:, blocks["LEFT_HAND"]] == 0)
    assert np.all(feats[:, blocks["FLAGS"]] == [0, 1])
    assert np.any(feats[:, blocks["RIGHT_HAND"]] != 0)


def test_pose_is_relative_to_shoulders():
    feats = frame_features(make_raw(3), LAYOUT, CFG)
    pose = feats[:, LAYOUT.blocks["POSE"]].reshape(3, -1, 3)
    left_sh, right_sh = pose[:, 1], pose[:, 2]
    assert np.allclose((left_sh + right_sh) / 2, 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(left_sh[:, :2] - right_sh[:, :2], axis=-1), 1.0)


def test_missing_shoulders_carry_forward_and_back_fill():
    pose = make_raw(5).pose
    pose[1, :, 3] = 0.0
    pose[3, 11:13, 3] = 0.0
    pose[2, 11, :3] = [0.7, 0.4, 0.0]
    pose[0, :, 3] = 0.0
    center, scale = body_reference(pose, 0.5, EPS)
    expected_center = (pose[:, 11, :3] + pose[:, 12, :3]) / 2
    assert np.allclose(center[0], expected_center[2])
    assert np.allclose(center[1], expected_center[2])
    assert np.allclose(center[3], expected_center[2])
    assert np.allclose(center[4], expected_center[4])
    assert scale[3] == pytest.approx(scale[2])


def test_no_shoulders_anywhere_falls_back_to_nose():
    pose = make_raw(3).pose
    pose[:, 11:13, 3] = 0.0
    center, scale = body_reference(pose, 0.5, EPS)
    assert np.allclose(center, pose[:, 0, :3])
    assert np.all(scale == 1.0)


def test_flags_are_not_normalized():
    feats = frame_features(make_raw(2), LAYOUT, CFG)
    assert set(np.unique(feats[:, LAYOUT.blocks["FLAGS"]])) <= {0.0, 1.0}


def test_trim_removes_frames_without_hands_only_at_the_ends():
    raw = make_raw(10)
    for t in (0, 1, 5, 9):
        raw.left_hand[t] = np.nan
        raw.right_hand[t] = np.nan
    trimmed = trim_to_hands(frame_features(raw, LAYOUT, CFG), LAYOUT)
    assert len(trimmed) == 7


def test_resample_length_and_endpoints():
    seq = RNG.normal(size=(17, 4)).astype(np.float32)
    out = resample(seq, 30)
    assert out.shape == (30, 4)
    assert out.dtype == np.float32
    assert np.allclose(out[0], seq[0])
    assert np.allclose(out[-1], seq[-1])


def test_resample_is_linear_in_time():
    seq = np.arange(5, dtype=np.float32)[:, None] * 2.0
    out = resample(seq, 9)
    assert np.allclose(out[:, 0], np.arange(9))


def test_resample_identity_and_single_frame():
    seq = RNG.normal(size=(30, 3)).astype(np.float32)
    assert np.allclose(resample(seq, 30), seq)
    assert np.allclose(resample(seq[:1], 30), np.repeat(seq[:1], 30, axis=0))


def test_build_sequence_shape_and_short_clip_rejection():
    seq = build_sequence(make_raw(40), LAYOUT, CFG)
    assert seq.shape == (CFG["temporal"]["seq_len"], LAYOUT.size)
    assert seq.dtype == np.float32
    short = make_raw(40)
    short.left_hand[5:] = np.nan
    short.right_hand[5:] = np.nan
    assert build_sequence(short, LAYOUT, CFG) is None
