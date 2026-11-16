"""Top-k word predictions for one sign, from a feature sequence or from raw landmarks.

The live app, the replay script and the HTTP API all go through Predictor, so a clip gives the
same answer whichever way it arrives.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.evaluate import predict_proba
from src.features import FeatureLayout, build_sequence
from src.landmarks import FrameLandmarks, RawLandmarks
from src.models import load_bundle

TopWords = list[tuple[str, float]]


class Predictor:
    """A saved model bundle together with the feature layout it was trained on."""

    def __init__(self, bundle: dict[str, Any], cfg: dict[str, Any]):
        self.bundle = bundle
        self.cfg = cfg
        self.layout = FeatureLayout.from_config(cfg)
        if not np.array_equal(bundle["columns"], self.layout.columns(*bundle["groups"])):
            raise ValueError("model was trained with a different feature layout than config.yaml describes")

    @classmethod
    def from_path(cls, model_path: str | Path, cfg: dict[str, Any]) -> Predictor:
        return cls(load_bundle(Path(model_path)), cfg)

    @property
    def words(self) -> list[str]:
        return list(self.bundle["words"])

    @property
    def input_shape(self) -> tuple[int, int]:
        """(seq_len, features) of one normalized sequence, before the model selects its columns."""
        return self.cfg["temporal"]["seq_len"], self.layout.size

    def sequence(self, frames: list[FrameLandmarks], fps: float) -> np.ndarray | None:
        """The (seq_len, features) input for these frames, or None if too few frames show a hand."""
        return build_sequence(RawLandmarks.from_frames(frames, fps), self.layout, self.cfg)

    def predict_sequence(self, sequence: np.ndarray, k: int = 3) -> TopWords:
        """The k most likely (word, probability) pairs for one normalized sequence, best first."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape != self.input_shape:
            raise ValueError(f"expected a sequence of shape {self.input_shape}, got {seq.shape}")
        proba = predict_proba(self.bundle, seq[None])[0]
        best = np.argsort(proba)[::-1][:k]
        return [(self.words[i], float(proba[i])) for i in best]

    def predict_frames(self, frames: list[FrameLandmarks], fps: float, k: int = 3) -> TopWords | None:
        seq = self.sequence(frames, fps)
        return None if seq is None else self.predict_sequence(seq, k)


@lru_cache(maxsize=1)
def default_predictor(config_path: str = "config.yaml") -> Predictor:
    cfg = load_config(config_path)
    return Predictor.from_path(cfg["live"]["model"], cfg)


def predict(sequence: np.ndarray, k: int = 3) -> TopWords:
    """Top-k words for one normalized (seq_len, features) array, using the model named in config.yaml."""
    return default_predictor().predict_sequence(sequence, k)
