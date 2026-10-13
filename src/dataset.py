"""Processed dataset loading and the fixed train/val/test split.

INCLUDE has no signer IDs. A signer records many words in one sitting, so the camera file
numbers of that sitting are close together across words (GOOD 5158-5160, BAD 5161-5163, ...).
Clips are grouped into sittings by those numbers and whole sittings go to one split.
Grouping per word alone was not enough: the model then recognised the person, and e.g. every
GOOD test clip was predicted as BAD because the same signer's BAD clips were in training.
This makes the test set sitting-independent, not signer-independent (a signer can return in
another sitting); own recordings provide the signer-independent test.

    python -m src.dataset            # create data/processed/split.json (refuses to overwrite)
    python -m src.dataset --force    # recreate it
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config


@dataclass
class Dataset:
    X: np.ndarray  # (N, seq_len, F)
    y: np.ndarray  # (N,)
    words: list[str]
    paths: np.ndarray
    signers: np.ndarray

    def subset(self, idx: np.ndarray) -> Dataset:
        return Dataset(self.X[idx], self.y[idx], self.words, self.paths[idx], self.signers[idx])


def load_dataset(path: str | Path) -> Dataset:
    with np.load(path) as d:
        return Dataset(d["X"], d["y"], [str(w) for w in d["words"]], d["paths"], d["signers"])


def file_number(path: str) -> int:
    """Camera counter of an INCLUDE file. A few names have typos (MVI_0057_, MVI_005_6), so keep only digits."""
    return int(re.sub(r"\D", "", Path(path).stem))


def _runs(idx: np.ndarray, nums: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Split clip indices into runs of camera numbers whose neighbours differ by at most max_gap."""
    order = idx[np.argsort(nums[idx], kind="stable")]
    breaks = np.flatnonzero(np.diff(nums[order]) > max_gap) + 1
    return np.split(order, breaks)


def sitting_groups(paths: np.ndarray, max_gap: int, max_clips: int, fallback_gap: int) -> np.ndarray:
    """Sitting id per clip, from camera numbers across all words.

    Runs larger than max_clips (cameras with overlapping counters chained together) are split
    again with the tighter fallback_gap. Merging two real sittings is harmless; splitting one
    would leak a signer across splits, so the thresholds err towards larger groups.
    """
    nums = np.array([file_number(p) for p in paths])
    groups = np.empty(len(paths), dtype=np.int64)
    gid = 0
    for run in _runs(np.arange(len(paths)), nums, max_gap):
        for part in (_runs(run, nums, fallback_gap) if len(run) > max_clips else [run]):
            groups[part] = gid
            gid += 1
    return groups


def _greedy_take(order: list[int], counts: np.ndarray, target: np.ndarray) -> list[int]:
    """Take groups in order whenever that brings the per-word clip counts closer to target."""
    taken, current = [], np.zeros_like(target)
    for g in order:
        if np.abs(current + counts[g] - target).sum() < np.abs(current - target).sum():
            taken.append(g)
            current = current + counts[g]
    return taken


def make_split(labels: np.ndarray, groups: np.ndarray, test_fraction: float, val_fraction: float,
               seed: int, max_attempts: int = 2000) -> dict[str, np.ndarray]:
    """Train/val/test indices made of whole groups.

    Groups span several words, so a split cannot be stratified word by word. Instead, many
    seeded random orders of the groups are tried; each fills test, then val, greedily towards
    the per-word targets. The order whose split covers every word in every part and lands
    closest to the targets wins.
    """
    group_ids, group_of = np.unique(groups, return_inverse=True)
    n_words = labels.max() + 1
    counts = np.zeros((len(group_ids), n_words))
    np.add.at(counts, (group_of, labels), 1)
    total = counts.sum(axis=0)

    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(max_attempts):
        order = [int(g) for g in rng.permutation(len(group_ids))]
        test = _greedy_take(order, counts, test_fraction * total)
        rest = [g for g in order if g not in set(test)]
        val = _greedy_take(rest, counts, val_fraction * counts[rest].sum(axis=0))
        train = [g for g in rest if g not in set(val)]
        parts = {"train": train, "val": val, "test": test}
        per_word = {k: counts[v].sum(axis=0) for k, v in parts.items()}
        if any((c == 0).any() for c in per_word.values()):
            continue
        score = (np.abs(per_word["test"] - test_fraction * total).sum()
                 + np.abs(per_word["val"] - val_fraction * (total - per_word["test"])).sum())
        if score < best_score:
            best, best_score = parts, score
    if best is None:
        raise ValueError("no split covers every word in train, val and test; loosen the grouping thresholds")
    return {k: np.flatnonzero(np.isin(group_of, v)) for k, v in best.items()}


def save_split(path: Path, split: dict[str, np.ndarray], data: Dataset, groups: np.ndarray,
               meta: dict[str, Any]) -> None:
    payload = dict(meta)
    for name, idx in split.items():
        payload[name] = {"indices": idx.tolist(), "paths": data.paths[idx].tolist(), "groups": groups[idx].tolist()}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_split(path: str | Path, data: Dataset) -> dict[str, np.ndarray]:
    """Read split.json and check it still matches the dataset it was made for."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = {}
    for name in ("train", "val", "test"):
        idx = np.asarray(payload[name]["indices"], dtype=np.int64)
        if data.paths[idx].tolist() != payload[name]["paths"]:
            raise ValueError(f"{path} does not match the dataset; rerun `python -m src.dataset --force`")
        split[name] = idx
    return split


def describe_split(split: dict[str, np.ndarray], data: Dataset, groups: np.ndarray) -> None:
    n_classes = len(data.words)
    for name, idx in split.items():
        counts = np.bincount(data.y[idx], minlength=n_classes)
        print(f"{name:5s} {len(idx):4d} clips, {len(np.unique(groups[idx])):3d} sittings, "
              f"per word min {counts.min()} max {counts.max()}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(groups[split[a]]) & set(groups[split[b]])
        assert not shared, f"sittings shared between {a} and {b}: {shared}"
    print("no sitting appears in more than one split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed_dir"])
    data_path, split_path = processed / "include.npz", processed / "split.json"
    if split_path.exists() and not args.force:
        raise SystemExit(f"{split_path} exists. Every model must use the same split; pass --force to recreate it.")

    data = load_dataset(data_path)
    sc = cfg["split"]
    groups = sitting_groups(data.paths, sc["sitting_max_gap"], sc["max_sitting_clips"], sc["fallback_gap"])
    print(f"{len(np.unique(groups))} sittings, {np.bincount(groups).min()}-{np.bincount(groups).max()} clips each")
    split = make_split(data.y, groups, sc["test_fraction"], sc["val_fraction"], cfg["seed"], sc["max_attempts"])
    describe_split(split, data, groups)
    save_split(split_path, split, data, groups, {"source": data_path.as_posix(), "seed": cfg["seed"], **sc})
    print(f"wrote {split_path}")


if __name__ == "__main__":
    main()
