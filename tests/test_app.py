import csv

import numpy as np

from src.app import KEY_BACKSPACE, LiveApp, Practice
from src.features import FeatureLayout
from src.landmarks import FrameLandmarks
from src.recognizer import Prediction
from src.segmenter import Segment

CFG = {
    "segmenter": {"start_frames": 3, "preroll_frames": 2, "end_absent_frames": 4, "end_still_frames": 5,
                  "max_frames": 50, "motion_threshold": 0.01},
    "live": {"min_confidence": 0.5},
}
IMAGES = [np.zeros((48, 64, 3), np.uint8)] * 5


class StubRecognizer:
    """Returns queued predictions instead of running a model."""

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.bundle = {"groups": ["HANDS", "POSE", "FLAGS"]}
        self.layout = FeatureLayout((0, 11, 12), (1,))

    def predict(self, frames, fps):
        return self.predictions.pop(0)


def pred(word, p):
    return Prediction([(word, p), ("other", 1 - p)])


def finish(app):
    empty = FrameLandmarks(None, None, None, None)
    return app._finish(Segment([(empty, IMAGES[0])] * 5, "absent"), 25.0)


def test_confident_words_build_the_sentence_and_keys_edit_it():
    app = LiveApp(CFG, StubRecognizer([pred("hello", 0.9), pred("you", 0.3), pred("friend", 0.8)]))
    for _ in range(3):
        finish(app)
    assert app.sentence == ["hello", "friend"]
    assert app.last.word == "friend"
    assert app.handle_key(KEY_BACKSPACE) and app.sentence == ["hello"]
    assert app.handle_key(ord("c")) and app.sentence == []
    assert app.handle_key(ord("q")) is False


def test_too_short_segment_adds_nothing():
    app = LiveApp(CFG, StubRecognizer([None]))
    assert finish(app) is None and app.sentence == [] and app.last is None


def test_practice_saves_numbered_takes_and_moves_to_next_word(tmp_path):
    practice = Practice(["hello", "you"], 2, "mansi", tmp_path)
    (tmp_path / "hello").mkdir()
    (tmp_path / "hello" / "mansi_1.mp4").write_bytes(b"from an earlier session")
    app = LiveApp(CFG, StubRecognizer([pred("hello", 0.9), pred("you", 0.6), pred("you", 0.7)]), practice)
    finish(app)
    finish(app)
    assert sorted(p.name for p in (tmp_path / "hello").glob("*.mp4")) == ["mansi_1.mp4", "mansi_2.mp4", "mansi_3.mp4"]
    assert practice.target == "you"
    finish(app)
    assert app.sentence == []  # practice mode does not build a sentence
    with practice.log_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["word"], r["file"], r["predicted"]) for r in rows] == [
        ("hello", "hello/mansi_2.mp4", "hello"), ("hello", "hello/mansi_3.mp4", "you"), ("you", "you/mansi_1.mp4", "you")]


def test_practice_redo_deletes_the_take_and_asks_again(tmp_path):
    practice = Practice(["hello", "you"], 1, "mansi", tmp_path)
    app = LiveApp(CFG, StubRecognizer([pred("hello", 0.9)]), practice)
    finish(app)
    assert practice.target == "you"
    app.handle_key(ord("r"))
    assert practice.target == "hello"
    assert not (tmp_path / "hello" / "mansi_1.mp4").exists()
    with practice.log_path.open(encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []


def test_practice_skip_and_completion(tmp_path):
    practice = Practice(["hello", "you"], 1, "mansi", tmp_path)
    app = LiveApp(CFG, StubRecognizer([]), practice)
    app.handle_key(ord("n"))
    app.handle_key(ord("n"))
    assert practice.done


def test_practice_does_not_save_too_short_segments(tmp_path):
    practice = Practice(["hello"], 1, "mansi", tmp_path)
    app = LiveApp(CFG, StubRecognizer([None]), practice)
    finish(app)
    assert practice.saved == [] and not (tmp_path / "hello").exists()
