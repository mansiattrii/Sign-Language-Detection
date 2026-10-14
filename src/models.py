"""Model builders. Every builder takes the input feature size as a parameter, so the ablation
can reuse the exact same architecture on any subset of feature groups."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier

RECURRENT_CELLS = {"lstm": keras.layers.LSTM, "gru": keras.layers.GRU}


def summary_features(X: np.ndarray) -> np.ndarray:
    """Per-clip mean and standard deviation of every feature over time: (N, T, F) -> (N, 2F).

    This throws away temporal order, which is exactly what the baseline is meant to measure.
    """
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


def build_random_forest(rf_cfg: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_features=rf_cfg["max_features"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1,
    )


def build_recurrent(n_features: int, n_classes: int, seq_len: int, rec_cfg: dict[str, Any],
                    cell: str = "lstm") -> keras.Model:
    """Two stacked recurrent layers, then a small dense head.

    With the default config and cell="lstm": LSTM(128, return_sequences) -> Dropout(0.3) ->
    LSTM(64) -> Dropout(0.3) -> Dense(64, relu) -> Dense(n_classes, softmax). cell="gru" swaps in
    GRU layers of the same sizes for the comparison.
    """
    layer = RECURRENT_CELLS[cell]
    first, second = rec_cfg["units"]
    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_features)),
        layer(first, return_sequences=True),
        keras.layers.Dropout(rec_cfg["dropout"]),
        layer(second),
        keras.layers.Dropout(rec_cfg["dropout"]),
        keras.layers.Dense(rec_cfg["dense_units"], activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name=cell)
    model.compile(optimizer=keras.optimizers.Adam(rec_cfg["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_bundle(bundle: dict[str, Any], models_dir: Path) -> Path:
    """Save a trained model and its metadata as models/<name>.joblib.

    Keras models are saved separately as models/<name>.keras, since they do not pickle reliably.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(bundle)
    if bundle["kind"] in RECURRENT_CELLS:
        keras_path = models_dir / f"{bundle['name']}.keras"
        bundle["model"].save(keras_path)
        meta["model"] = keras_path.name
    out = models_dir / f"{bundle['name']}.joblib"
    joblib.dump(meta, out)
    return out


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle["kind"] in RECURRENT_CELLS:
        bundle["model"] = keras.models.load_model(Path(path).parent / bundle["model"], compile=False)
    return bundle
