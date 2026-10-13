"""Train a model on the fixed split and save it to models/.

    python -m src.train --model rf
    python -m src.train --model rf --groups HANDS FLAGS --name rf_hands
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.augment import Augmenter
from src.config import load_config
from src.dataset import Dataset, load_dataset, load_split
from src.features import FeatureLayout
from src.models import build_random_forest, summary_features

log = logging.getLogger("train")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_training_data(cfg: dict[str, Any]) -> tuple[Dataset, dict[str, np.ndarray]]:
    processed = Path(cfg["paths"]["processed_dir"])
    data = load_dataset(processed / "include.npz")
    return data, load_split(processed / "split.json", data)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, cfg: dict[str, Any], layout: FeatureLayout,
                        columns: np.ndarray, seed: int) -> Any:
    copies = cfg["random_forest"]["augment_copies"]
    if copies:
        augmenter = Augmenter(layout, cfg, seed)
        extra = [augmenter.batch(X_train)[:, :, columns] for _ in range(copies)]
        X_fit = np.concatenate([X_train[:, :, columns], *extra])
        y_fit = np.concatenate([y_train] * (copies + 1))
    else:
        X_fit, y_fit = X_train[:, :, columns], y_train
    model = build_random_forest(cfg["random_forest"], seed)
    model.fit(summary_features(X_fit), y_fit)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["rf"], required=True)
    parser.add_argument("--groups", nargs="+", default=None, help="feature groups, default from config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="output name, default is the model type")
    parser.add_argument("--augment-copies", type=int, default=None,
                        help="RF only: augmented copies per training clip, overrides config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.augment_copies is not None:
        cfg["random_forest"]["augment_copies"] = args.augment_copies
    seed = cfg["seed"] if args.seed is None else args.seed
    groups = args.groups or cfg["feature_groups"]
    name = args.name or args.model
    set_seeds(seed)

    layout = FeatureLayout.from_config(cfg)
    columns = layout.columns(*groups)
    data, split = load_training_data(cfg)
    train, val = data.subset(split["train"]), data.subset(split["val"])
    log.info("model %s | groups %s (%d features) | seed %d | train %d, val %d",
             args.model, groups, len(columns), seed, len(train.y), len(val.y))

    model = train_random_forest(train.X, train.y, cfg, layout, columns, seed)
    val_acc = float((model.predict(summary_features(val.X[:, :, columns])) == val.y).mean())
    log.info("validation accuracy %.3f", val_acc)

    out = Path(cfg["paths"]["models_dir"]) / f"{name}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": args.model, "model": model, "groups": groups, "columns": columns, "words": data.words,
                 "seed": seed, "name": name}, out)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
