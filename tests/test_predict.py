import numpy as np
import pytest

from src.config import load_config
from src.features import build_sequence
from src.predict import Predictor, predict
from src.recognizer import Recognizer
from tests.test_features import LAYOUT, make_raw

CFG = load_config("config.yaml")
MODEL = CFG["live"]["model"]


def test_predict_gives_top3_in_descending_order():
    top = predict(build_sequence(make_raw(40), LAYOUT, CFG))
    assert len(top) == 3
    assert all(word in CFG["words"] for word, _ in top)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert 0 < sum(probs) <= 1 + 1e-6


def test_predictor_reports_its_input_shape_and_rejects_others():
    predictor = Predictor.from_path(MODEL, CFG)
    assert predictor.input_shape == (30, 308)
    with pytest.raises(ValueError, match=r"\(30, 308\)"):
        predictor.predict_sequence(np.zeros((29, 308), np.float32))
    with pytest.raises(ValueError, match=r"\(30, 308\)"):
        predictor.predict_sequence(np.zeros((30, 149), np.float32))


def test_frames_and_sequence_paths_agree_and_recognizer_reuses_them():
    predictor = Predictor.from_path(MODEL, CFG)
    raw = make_raw(40)
    from_frames = predictor.predict_frames(raw.frames(), raw.fps)
    from_sequence = predictor.predict_sequence(build_sequence(raw, LAYOUT, CFG))
    assert from_frames == from_sequence
    recognizer = Recognizer.from_path(MODEL, CFG)
    assert isinstance(recognizer, Predictor)
    assert recognizer.predict(raw.frames(), raw.fps).top == from_frames
    assert predictor.predict_frames(make_raw(40, left=False, right=False).frames(), 25.0) is None
