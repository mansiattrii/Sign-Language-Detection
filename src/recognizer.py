"""Turns the landmarks of one segmented sign into word probabilities.

Uses build_sequence, the exact function that built the training data, so a live sign goes
through the same normalization, trimming and resampling as the INCLUDE clips.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate import predict_proba
from src.features import FeatureLayout, build_sequence
from src.landmarks import FrameLandmarks, RawLandmarks
from src.models import load_bundle


@dataclass
class Prediction:
    top: list[tuple[str, float]]  # (word, probability), most likely first

    @property
    def word(self) -> str:
        return self.top[0][0]

    @property
    def confidence(self) -> float:
        return self.top[0][1]


class Recognizer:
    def __init__(self, bundle: dict[str, Any], cfg: dict[str, Any]):
        self.bundle = bundle
        self.cfg = cfg
        self.layout = FeatureLayout.from_config(cfg)
        if not np.array_equal(bundle["columns"], self.layout.columns(*bundle["groups"])):
            raise ValueError("model was trained with a different feature layout than config.yaml describes")

    @classmethod
    def from_path(cls, model_path: Path, cfg: dict[str, Any]) -> Recognizer:
        return cls(load_bundle(model_path), cfg)

    def sequence(self, frames: list[FrameLandmarks], fps: float) -> np.ndarray | None:
        """The (seq_len, features) input for these frames, or None if too few frames show a hand."""
        return build_sequence(RawLandmarks.from_frames(frames, fps), self.layout, self.cfg)

    def predict(self, frames: list[FrameLandmarks], fps: float, k: int = 3) -> Prediction | None:
        seq = self.sequence(frames, fps)
        if seq is None:
            return None
        proba = predict_proba(self.bundle, seq[None])[0]
        best = np.argsort(proba)[::-1][:k]
        return Prediction([(self.bundle["words"][i], float(proba[i])) for i in best])
