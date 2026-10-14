"""Train a model on the fixed split and save it to models/.

    python -m src.train --model lstm
    python -m src.train --model gru
    python -m src.train --model rf
    python -m src.train --model lstm --groups HANDS POSE FLAGS --seed 43 --name lstm_hp_s43
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from src.augment import Augmenter
from src.config import load_config
from src.dataset import Dataset, load_dataset, load_split
from src.features import FeatureLayout
from src.models import RECURRENT_CELLS, build_random_forest, build_recurrent, save_bundle, summary_features

log = logging.getLogger("train")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_training_data(cfg: dict[str, Any]) -> tuple[Dataset, dict[str, np.ndarray]]:
    processed = Path(cfg["paths"]["processed_dir"])
    data = load_dataset(processed / "include.npz")
    return data, load_split(processed / "split.json", data)


class AugmentedBatches(keras.utils.PyDataset):
    """Shuffled training batches, freshly augmented every time they are drawn.

    Augmentation runs on all columns, because it needs the full layout to find each block; the
    columns of the chosen feature groups are selected afterwards.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray, batch_size: int,
                 augmenter: Augmenter | None, seed: int):
        super().__init__()
        self.X, self.y, self.columns = X, y, columns
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(y))

    def __len__(self) -> int:
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        Xb = self.X[idx]
        if self.augmenter is not None:
            Xb = self.augmenter.batch(Xb)
        return Xb[:, :, self.columns], self.y[idx]

    def on_epoch_end(self) -> None:
        self.order = self.rng.permutation(len(self.y))


def class_weights(y: np.ndarray, n_classes: int, ratio: float) -> dict[int, float] | None:
    """Balanced weights n / (k * count_c) if the largest class has at least `ratio` times the
    clips of the smallest, otherwise None."""
    counts = np.bincount(y, minlength=n_classes)
    if counts.max() < ratio * counts.min():
        return None
    return {c: len(y) / (n_classes * counts[c]) for c in range(n_classes) if counts[c]}


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


def train_recurrent(train: Dataset, val: Dataset, cfg: dict[str, Any], layout: FeatureLayout,
                    columns: np.ndarray, seed: int, cell: str) -> tuple[keras.Model, dict[str, Any]]:
    """Fit with early stopping on the configured validation metric, restoring the best epoch's weights."""
    rec = cfg["recurrent"]
    n_classes = len(train.words)
    model = build_recurrent(len(columns), n_classes, train.X.shape[1], rec, cell)
    augmenter = Augmenter(layout, cfg, seed) if rec["augment"] else None
    batches = AugmentedBatches(train.X, train.y, columns, rec["batch_size"], augmenter, seed)
    weights = class_weights(train.y, n_classes, rec["class_weight_ratio"])
    monitor = rec["monitor"]
    stop = keras.callbacks.EarlyStopping(monitor=monitor, patience=rec["patience"], restore_best_weights=True,
                                         mode="min" if monitor.endswith("loss") else "max")
    history = model.fit(batches, validation_data=(val.X[:, :, columns], val.y), epochs=rec["max_epochs"],
                        class_weight=weights, callbacks=[stop], verbose=0)
    curve = np.asarray(history.history[monitor])
    best = int(np.argmin(curve) if monitor.endswith("loss") else np.argmax(curve))
    info = {"epochs_run": len(curve), "best_epoch": best + 1, "monitor": monitor,
            "class_weights": weights is not None, "augment": augmenter is not None}
    return model, info


def fit(kind: str, data: Dataset, split: dict[str, np.ndarray], cfg: dict[str, Any], groups: list[str],
        seed: int, name: str) -> dict[str, Any]:
    """Train one model and return its bundle: the model plus everything needed to use it again."""
    set_seeds(seed)
    layout = FeatureLayout.from_config(cfg)
    columns = layout.columns(*groups)
    train, val = data.subset(split["train"]), data.subset(split["val"])
    log.info("%s | %s | groups %s (%d features) | seed %d | train %d, val %d",
             name, kind, groups, len(columns), seed, len(train.y), len(val.y))

    info: dict[str, Any] = {}
    if kind == "rf":
        model = train_random_forest(train.X, train.y, cfg, layout, columns, seed)
        val_pred = model.predict(summary_features(val.X[:, :, columns]))
    else:
        model, info = train_recurrent(train, val, cfg, layout, columns, seed, kind)
        val_pred = model.predict(val.X[:, :, columns], verbose=0).argmax(axis=1)
        log.info("stopped after %d epochs, best epoch %d", info["epochs_run"], info["best_epoch"])
    info["val_accuracy"] = float((val_pred == val.y).mean())
    log.info("validation accuracy %.3f", info["val_accuracy"])
    return {"kind": kind, "model": model, "groups": list(groups), "columns": columns, "words": data.words,
            "seed": seed, "name": name, "train_info": info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["rf", *RECURRENT_CELLS], required=True)
    parser.add_argument("--groups", nargs="+", default=None, help="feature groups, default from config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="output name, default is the model type")
    parser.add_argument("--augment-copies", type=int, default=None,
                        help="RF only: augmented copies per training clip, overrides config")
    parser.add_argument("--no-augment", action="store_true", help="LSTM/GRU only: train without augmentation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.augment_copies is not None:
        cfg["random_forest"]["augment_copies"] = args.augment_copies
    if args.no_augment:
        cfg["recurrent"]["augment"] = False
    seed = cfg["seed"] if args.seed is None else args.seed
    groups = args.groups or cfg["feature_groups"]

    data, split = load_training_data(cfg)
    bundle = fit(args.model, data, split, cfg, groups, seed, args.name or args.model)
    log.info("saved %s", save_bundle(bundle, Path(cfg["paths"]["models_dir"])))


if __name__ == "__main__":
    main()
