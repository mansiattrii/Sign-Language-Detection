"""HTTP inference service for the trained word recognizer.

    uvicorn api.app:app --host 0.0.0.0 --port 8000

MODEL_PATH and CONFIG_PATH override the model named in config.yaml and the config file itself.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from src.config import load_config
from src.landmarks import N_FACE, N_HAND, N_POSE, FrameLandmarks
from src.predict import Predictor

PART_SHAPES = {"left_hand": (N_HAND, 3), "right_hand": (N_HAND, 3), "pose": (N_POSE, 4), "face": (N_FACE, 3)}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config(os.environ.get("CONFIG_PATH", "config.yaml"))
    model_path = Path(os.environ.get("MODEL_PATH", cfg["live"]["model"]))
    app.state.predictor = Predictor.from_path(model_path, cfg)
    app.state.model_name = model_path.name
    yield


app = FastAPI(title="ISL word recognizer", version="1.0.0", lifespan=lifespan)


class SequenceRequest(BaseModel):
    sequence: list[list[float]] = Field(description="One normalized clip of shape (seq_len, features); see /health.")


class Frame(BaseModel):
    """Holistic landmarks of one frame, as src.landmarks.FrameLandmarks stores them: x and z are
    multiplied by the image aspect ratio, and a part Holistic did not detect is null."""

    left_hand: list[list[float]] | None = Field(None, description="(21, 3) x, y, z")
    right_hand: list[list[float]] | None = Field(None, description="(21, 3) x, y, z")
    pose: list[list[float]] | None = Field(None, description="(33, 4) x, y, z, visibility")
    face: list[list[float]] | None = Field(None, description="(468, 3) x, y, z; unused by the final model")

    @field_validator("left_hand", "right_hand", "pose", "face")
    @classmethod
    def check_shape(cls, value: list[list[float]] | None, info: ValidationInfo) -> list[list[float]] | None:
        if value is None:
            return None
        expected = PART_SHAPES[info.field_name]
        shape = (len(value), len(value[0]) if value else 0)
        if shape != expected or any(len(row) != expected[1] for row in value):
            raise ValueError(f"must have shape {expected}, got {shape}")
        return value

    def to_landmarks(self) -> FrameLandmarks:
        def part(value: list[list[float]] | None) -> np.ndarray | None:
            return None if value is None else np.asarray(value, dtype=np.float32)

        return FrameLandmarks(part(self.left_hand), part(self.right_hand), part(self.pose), part(self.face))


class LandmarksRequest(BaseModel):
    frames: list[Frame] = Field(min_length=1)
    fps: float = Field(25.0, gt=0)


class WordProbability(BaseModel):
    word: str
    probability: float


class PredictionResponse(BaseModel):
    predictions: list[WordProbability]


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def _response(top: list[tuple[str, float]]) -> PredictionResponse:
    return PredictionResponse(predictions=[WordProbability(word=w, probability=p) for w, p in top])


@app.get("/health")
def health(request: Request) -> dict:
    predictor = _predictor(request)
    seq_len, n_features = predictor.input_shape
    return {"status": "ok", "model": request.app.state.model_name, "num_classes": len(predictor.words),
            "seq_len": seq_len, "num_features": n_features}


@app.post("/predict")
def predict(body: SequenceRequest, request: Request) -> PredictionResponse:
    try:
        return _response(_predictor(request).predict_sequence(body.sequence))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/landmarks")
def predict_landmarks(body: LandmarksRequest, request: Request) -> PredictionResponse:
    predictor = _predictor(request)
    top = predictor.predict_frames([frame.to_landmarks() for frame in body.frames], body.fps)
    if top is None:
        min_frames = predictor.cfg["temporal"]["min_hand_frames"]
        raise HTTPException(status_code=422,
                            detail=f"too few frames with a visible hand: at least {min_frames} are needed")
    return _response(top)
