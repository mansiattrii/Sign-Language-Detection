"""Model builders. Every builder takes the input feature size as a parameter, so the ablation
can reuse the exact same architecture on any subset of feature groups."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


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
