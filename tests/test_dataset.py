import numpy as np
import pytest

from src.dataset import Dataset, file_number, load_split, make_split, save_split, sitting_groups


def synthetic(n_words: int = 6, sittings: int = 8, takes: int = 3):
    """Each sitting records every word in a row, like INCLUDE: sitting s, word w, take k."""
    paths, labels = [], []
    for s in range(sittings):
        for w in range(n_words):
            for k in range(takes):
                paths.append(f"data/raw/w{w}/MVI_{1000 * s + takes * w + k:04d}.MOV")
                labels.append(w)
    return np.array(paths), np.array(labels)


def test_file_number_handles_typos():
    assert file_number("data/raw/good_night/MVI_0057_.MOV") == 57
    assert file_number("data/raw/good_night/MVI_005_6.MOV") == 56


def test_sitting_spans_words_recorded_back_to_back():
    paths = np.array(["good/MVI_5158.MOV", "good/MVI_5160.MOV", "bad/MVI_5161.MOV", "bad/MVI_5163.MOV",
                      "good/MVI_5238.MOV", "bad/MVI_5241.MOV"])
    g = sitting_groups(paths, max_gap=10, max_clips=60, fallback_gap=5)
    assert len(set(g[:4])) == 1
    assert g[4] == g[5] and g[4] != g[0]


def test_oversized_sitting_is_split_at_smaller_gaps():
    paths = np.array([f"w/MVI_{n:04d}.MOV" for n in [1, 2, 3, 9, 10, 11, 17, 18, 19]])
    assert len(set(sitting_groups(paths, 10, 100, 5))) == 1
    assert len(set(sitting_groups(paths, 10, 4, 5))) == 3


def test_split_keeps_sittings_together_and_covers_every_word():
    paths, labels = synthetic()
    groups = sitting_groups(paths, 10, 60, 5)
    split = make_split(labels, groups, 0.2, 0.2, seed=42, max_attempts=200)
    sets = {k: set(groups[v]) for k, v in split.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sorted(np.concatenate(list(split.values()))) == list(range(len(labels)))
    for idx in split.values():
        assert set(labels[idx]) == set(labels)


def test_split_is_deterministic_for_a_seed():
    paths, labels = synthetic()
    groups = sitting_groups(paths, 10, 60, 5)
    a = make_split(labels, groups, 0.2, 0.2, 7, max_attempts=100)
    b = make_split(labels, groups, 0.2, 0.2, 7, max_attempts=100)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_split_fails_loudly_when_impossible():
    paths, labels = synthetic(sittings=2)
    groups = sitting_groups(paths, 10, 60, 5)
    with pytest.raises(ValueError):
        make_split(labels, groups, 0.2, 0.2, 0, max_attempts=50)


def test_load_split_rejects_a_changed_dataset(tmp_path):
    paths, labels = synthetic()
    data = Dataset(np.zeros((len(labels), 30, 4)), labels, [f"w{i}" for i in range(6)], paths, np.array([""] * len(labels)))
    groups = sitting_groups(paths, 10, 60, 5)
    split = make_split(labels, groups, 0.2, 0.2, 0, max_attempts=100)
    save_split(tmp_path / "split.json", split, data, groups, {"seed": 0})
    assert all(np.array_equal(load_split(tmp_path / "split.json", data)[k], split[k]) for k in split)
    shuffled = data.subset(np.random.default_rng(0).permutation(len(labels)))
    with pytest.raises(ValueError):
        load_split(tmp_path / "split.json", shuffled)
