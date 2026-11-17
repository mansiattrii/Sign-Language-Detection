import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.dataset import load_dataset
from src.features import build_sequence
from src.landmarks import FrameLandmarks, RawLandmarks
from src.models import predict_proba
from tests.conftest import CACHE, DATASET
from tests.test_features import LAYOUT, make_raw

CFG = load_config("config.yaml")
needs_data = pytest.mark.needs_data


def frame_json(frame: FrameLandmarks) -> dict:
    return {name: None if part is None else part.tolist()
            for name, part in (("left_hand", frame.left_hand), ("right_hand", frame.right_hand),
                               ("pose", frame.pose), ("face", frame.face))}


def landmarks_body(raw: RawLandmarks) -> dict:
    return {"fps": raw.fps, "frames": [frame_json(f) for f in raw.frames()]}


def detail_text(response) -> str:
    return str(response.json()["detail"])


def test_health_describes_the_loaded_model(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model": "lstm.joblib", "num_classes": 30, "seq_len": 30, "num_features": 308}


def test_predict_returns_top3_sorted_by_probability(client):
    sequence = build_sequence(make_raw(40), LAYOUT, CFG)
    response = client.post("/predict", json={"sequence": sequence.tolist()})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert all(p["word"] in CFG["words"] for p in predictions)
    probs = [p["probability"] for p in predictions]
    assert probs == sorted(probs, reverse=True)


@needs_data
def test_predict_matches_the_offline_model(client):
    data = load_dataset(DATASET)
    expected = predict_proba(client.app.state.predictor.bundle, data.X[:1])[0]
    best = response_top(client.post("/predict", json={"sequence": data.X[0].tolist()}))
    assert best["word"] == data.words[int(expected.argmax())]
    assert best["probability"] == pytest.approx(float(expected.max()), abs=1e-5)


def response_top(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()["predictions"][0]


def test_predict_rejects_wrong_feature_count(client):
    response = client.post("/predict", json={"sequence": np.zeros((30, 100)).tolist()})
    assert response.status_code == 422
    assert "(30, 308)" in detail_text(response) and "(30, 100)" in detail_text(response)


def test_predict_rejects_wrong_frame_count(client):
    response = client.post("/predict", json={"sequence": np.zeros((29, 308)).tolist()})
    assert response.status_code == 422
    assert "(30, 308)" in detail_text(response) and "(29, 308)" in detail_text(response)


def test_predict_rejects_ragged_and_non_numeric_input(client):
    ragged = [[0.0] * 308] * 29 + [[0.0] * 307]
    assert client.post("/predict", json={"sequence": ragged}).status_code == 422
    assert client.post("/predict", json={"sequence": [["x"] * 308] * 30}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_landmarks_endpoint_agrees_with_the_sequence_endpoint(client):
    raw = make_raw(40)
    via_landmarks = client.post("/predict/landmarks", json=landmarks_body(raw))
    via_sequence = client.post("/predict", json={"sequence": build_sequence(raw, LAYOUT, CFG).tolist()})
    assert via_landmarks.status_code == 200, via_landmarks.text
    assert via_landmarks.json() == via_sequence.json()


def test_landmarks_endpoint_works_without_face_points(client):
    body = landmarks_body(make_raw(40))
    for frame in body["frames"]:
        frame["face"] = None
    assert client.post("/predict/landmarks", json=body).status_code == 200


@needs_data
def test_landmarks_endpoint_matches_the_offline_model(client):
    data = load_dataset(DATASET)
    path = Path(data.paths[250])
    raw = RawLandmarks.load(CACHE / path.parent.name / f"{path.stem}.npz")
    expected = predict_proba(client.app.state.predictor.bundle, data.X[250:251])[0]
    best = response_top(client.post("/predict/landmarks", json=landmarks_body(raw)))
    assert best["word"] == data.words[int(expected.argmax())]
    assert best["probability"] == pytest.approx(float(expected.max()), abs=1e-5)


def test_landmarks_endpoint_rejects_wrong_point_counts(client):
    body = landmarks_body(make_raw(12))
    body["frames"][3]["left_hand"] = body["frames"][3]["left_hand"][:20]
    response = client.post("/predict/landmarks", json=body)
    assert response.status_code == 422
    assert "(21, 3)" in detail_text(response)
    body = landmarks_body(make_raw(12))
    body["frames"][0]["pose"] = [[0.1, 0.2, 0.3]] * 33
    response = client.post("/predict/landmarks", json=body)
    assert response.status_code == 422
    assert "(33, 4)" in detail_text(response)


def test_landmarks_endpoint_rejects_clips_with_too_few_hand_frames(client):
    body = landmarks_body(make_raw(20, left=False, right=False))
    response = client.post("/predict/landmarks", json=body)
    assert response.status_code == 422
    assert "hand" in detail_text(response)
    assert client.post("/predict/landmarks", json={"frames": []}).status_code == 422


def test_api_imports_without_the_extraction_and_training_stack():
    """The Docker image installs requirements-api.txt only, so these packages must not be hard imports."""
    code = ("import sys; "
            "sys.modules.update({m: None for m in ('mediapipe', 'cv2', 'sklearn', 'pandas', 'matplotlib')}); "
            "import api.app")
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
