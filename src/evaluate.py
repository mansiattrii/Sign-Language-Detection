"""Evaluate a saved model on the INCLUDE test split and, if present, on own recordings.

    python -m src.evaluate --model models/lstm.joblib
    python -m src.evaluate --model models/rf.joblib

Writes results/<name>/{metrics.json, per_class.csv, confused_pairs.csv, confusion_matrix.png},
and the same under results/<name>/own/ for data/processed/own.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score

from src.config import load_config
from src.dataset import Dataset, load_dataset, load_split
from src.models import load_bundle, predict_proba

SEQUENTIAL_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT_DARK, TEXT_LIGHT = "#1a1a19", "#ffffff"


def confused_pairs(cm: np.ndarray, words: list[str], k: int = 5) -> pd.DataFrame:
    """The k word pairs with the most mistakes in either direction."""
    rows = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if cm[i, j] + cm[j, i]:
                rows.append({"word_a": words[i], "word_b": words[j], "a_as_b": int(cm[i, j]),
                             "b_as_a": int(cm[j, i]), "total": int(cm[i, j] + cm[j, i])})
    df = pd.DataFrame(rows, columns=["word_a", "word_b", "a_as_b", "b_as_a", "total"])
    return df.sort_values(["total", "word_a"], ascending=[False, True]).head(k).reset_index(drop=True)


def plot_confusion(cm: np.ndarray, words: list[str], title: str, path: Path) -> None:
    """Rows are true words, normalized to the share of that word's clips; cells show raw counts."""
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(words)
    fig, ax = plt.subplots(figsize=(0.36 * n + 3, 0.36 * n + 2))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    mesh = ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=1, edgecolors="#ffffff", linewidth=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j + 0.5, i + 0.5, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color=TEXT_LIGHT if share[i, j] > 0.5 else TEXT_DARK)
    labels = [w.replace("_", " ") for w in words]
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted word")
    ax.set_ylabel("True word")
    ax.set_title(title, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Share of the true word's clips")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def scores(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Top-1, top-5 and macro F1. Macro F1 averages only over words that occur in y, so a test set
    without some word (e.g. own recordings) is not penalised for it."""
    labels = np.arange(proba.shape[1])
    pred = proba.argmax(axis=1)
    present = np.isin(labels, y)
    return {
        "top1_accuracy": float((pred == y).mean()),
        "top5_accuracy": float(top_k_accuracy_score(y, proba, k=5, labels=labels)),
        "macro_f1": float(f1_score(y, pred, labels=labels[present], average="macro", zero_division=0)),
    }


def evaluate(bundle: dict[str, Any], data: Dataset, out_dir: Path, title: str,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score a model on `data` and write the report files; `extra` is added to metrics.json."""
    words = bundle["words"]
    proba = predict_proba(bundle, data.X)
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(data.y, pred, labels=np.arange(len(words)))
    support = cm.sum(axis=1)
    per_class = np.divide(np.diag(cm), support, out=np.full(len(words), np.nan), where=support > 0)

    metrics = {"model": bundle["name"], "groups": bundle["groups"], "n_clips": int(len(data.y)),
               **scores(data.y, proba), **(extra or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"word": words, "clips": support, "correct": np.diag(cm), "accuracy": per_class}) \
        .to_csv(out_dir / "per_class.csv", index=False, float_format="%.3f")
    pairs = confused_pairs(cm, words)
    pairs.to_csv(out_dir / "confused_pairs.csv", index=False)
    plot_confusion(cm, words, f"{title} (top-1 {metrics['top1_accuracy']:.1%}, n={len(data.y)})",
                   out_dir / "confusion_matrix.png")

    print(f"\n== {title}")
    print(f"top-1 {metrics['top1_accuracy']:.3f} | top-5 {metrics['top5_accuracy']:.3f} | macro F1 {metrics['macro_f1']:.3f} "
          f"| {metrics['n_clips']} clips")
    print("most confused pairs:")
    print(pairs.to_string(index=False) if len(pairs) else "  none")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    bundle = load_bundle(args.model)
    processed = Path(cfg["paths"]["processed_dir"])
    results = Path(cfg["paths"]["results_dir"]) / bundle["name"]

    data = load_dataset(processed / "include.npz")
    split = load_split(processed / "split.json", data)
    suffix = "" if args.split == "test" else f"_{args.split}"
    evaluate(bundle, data.subset(split[args.split]), results / f"include{suffix}",
             f"{bundle['name']}: INCLUDE {args.split}")

    own_path = processed / "own.npz"
    if own_path.exists():
        own = load_dataset(own_path)
        # Own recordings may cover only some words; compare with INCLUDE test on the same words.
        test = data.subset(split["test"])
        same = test.subset(np.flatnonzero(np.isin(test.y, own.y)))
        reference = {"own_words": int(len(np.unique(own.y))), "signers": sorted(set(own.signers)),
                     "include_test_same_words_clips": int(len(same.y)),
                     "include_test_same_words": scores(same.y, predict_proba(bundle, same.X)) if len(same.y) else None}
        evaluate(bundle, own, results / "own", f"{bundle['name']}: own recordings", reference)
        if len(same.y):
            ref = reference["include_test_same_words"]
            print(f"INCLUDE test on the same {reference['own_words']} word(s), {len(same.y)} clips: "
                  f"top-1 {ref['top1_accuracy']:.3f} | top-5 {ref['top5_accuracy']:.3f} | macro F1 {ref['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
