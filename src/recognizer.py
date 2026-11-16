"""Turns the landmarks of one segmented sign into word probabilities for the live app."""
from __future__ import annotations

from dataclasses import dataclass

from src.landmarks import FrameLandmarks
from src.predict import Predictor


@dataclass
class Prediction:
    top: list[tuple[str, float]]  # (word, probability), most likely first

    @property
    def word(self) -> str:
        return self.top[0][0]

    @property
    def confidence(self) -> float:
        return self.top[0][1]


class Recognizer(Predictor):
    def predict(self, frames: list[FrameLandmarks], fps: float, k: int = 3) -> Prediction | None:
        top = self.predict_frames(frames, fps, k)
        return None if top is None else Prediction(top)
