from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.dataset import load_dataset
from src.evaluate import predict_proba
from src.landmarks import FrameLandmarks, RawLandmarks
from src.recognizer import Recognizer

MODEL = Path("models/lstm.joblib")
DATASET = Path("data/processed/include.npz")
CACHE = Path("data/processed/landmarks/include")

needs_data = pytest.mark.skipif(not (MODEL.exists() and DATASET.exists() and CACHE.exists()),
                                reason="needs the trained model and extracted INCLUDE data")


@needs_data
def test_live_path_matches_training_features_and_predictions():
    cfg = load_config()
    recognizer = Recognizer.from_path(MODEL, cfg)
    data = load_dataset(DATASET)
    for i in (0, 250, 600):
        path = Path(data.paths[i])
        raw = RawLandmarks.load(CACHE / path.parent.name / f"{path.stem}.npz")
        frames = raw.frames()
        np.testing.assert_allclose(recognizer.sequence(frames, raw.fps), data.X[i], atol=1e-6)
        expected = predict_proba(recognizer.bundle, data.X[i:i + 1])[0]
        pred = recognizer.predict(frames, raw.fps, k=3)
        assert pred.word == data.words[int(expected.argmax())]
        assert pred.confidence == pytest.approx(float(expected.max()), abs=1e-5)


@needs_data
def test_too_few_hand_frames_gives_no_prediction():
    recognizer = Recognizer.from_path(MODEL, load_config())
    empty = FrameLandmarks(None, None, None, None)
    assert recognizer.predict([empty] * 20, 30.0) is None
