import keras
import numpy as np
import pandas as pd
import pytest

from src.ablation import summarize
from src.models import build_recurrent, load_bundle, save_bundle
from src.train import AugmentedBatches, class_weights

REC_CFG = {"units": [128, 64], "dropout": 0.3, "dense_units": 64, "learning_rate": 1e-3}


@pytest.mark.parametrize("n_features", [128, 149, 308])
def test_recurrent_input_and_output_shapes(n_features):
    model = build_recurrent(n_features, 30, 30, REC_CFG)
    assert model.input_shape == (None, 30, n_features)
    assert model.output_shape == (None, 30)


def test_lstm_architecture_matches_spec():
    model = build_recurrent(308, 30, 30, REC_CFG)
    layers = [(type(layer).__name__, getattr(layer, "units", None)) for layer in model.layers]
    assert layers == [("LSTM", 128), ("Dropout", None), ("LSTM", 64), ("Dropout", None),
                      ("Dense", 64), ("Dense", 30)]
    assert model.layers[0].return_sequences and not model.layers[2].return_sequences
    assert model.layers[-1].activation.__name__ == "softmax"


def test_gru_swaps_only_the_recurrent_layers():
    model = build_recurrent(308, 30, 30, REC_CFG, cell="gru")
    assert [type(layer).__name__ for layer in model.layers] == ["GRU", "Dropout", "GRU", "Dropout", "Dense", "Dense"]


def test_class_weights_only_when_imbalanced():
    assert class_weights(np.repeat([0, 1, 2], [12, 13, 13]), 3, 1.5) is None
    weights = class_weights(np.repeat([0, 1], [10, 30]), 2, 1.5)
    assert weights == pytest.approx({0: 2.0, 1: 2 / 3})


def test_augmented_batches_select_columns_and_cover_every_clip():
    X = np.random.default_rng(0).normal(size=(10, 30, 8)).astype(np.float32)
    y = np.arange(10)
    columns = np.array([0, 2, 5])
    batches = AugmentedBatches(X, y, columns, batch_size=4, augmenter=None, seed=0)
    assert len(batches) == 3
    seen = []
    for i in range(len(batches)):
        Xb, yb = batches[i]
        assert Xb.shape[1:] == (30, 3)
        np.testing.assert_array_equal(Xb, X[yb][:, :, columns])
        seen.extend(yb)
    assert sorted(seen) == list(range(10))


def test_saved_recurrent_bundle_predicts_the_same(tmp_path):
    keras.utils.set_random_seed(0)
    model = build_recurrent(6, 4, 30, REC_CFG)
    bundle = {"kind": "lstm", "model": model, "groups": ["HANDS"], "columns": np.arange(6),
              "words": ["a", "b", "c", "d"], "seed": 0, "name": "tiny"}
    path = save_bundle(bundle, tmp_path)
    X = np.random.default_rng(1).normal(size=(3, 30, 6)).astype(np.float32)
    loaded = load_bundle(path)
    np.testing.assert_allclose(loaded["model"].predict(X, verbose=0), model.predict(X, verbose=0), rtol=1e-5)
    assert loaded["words"] == bundle["words"]


def test_summarize_mean_and_sample_std_in_config_order():
    df = pd.DataFrame({
        "config": ["all", "all", "all", "hands", "hands", "hands"],
        "n_features": [308] * 3 + [128] * 3,
        "val_top1": [0.5, 0.6, 0.7, 0.1, 0.2, 0.3],
        "test_top1": [0.4, 0.5, 0.6, 0.2, 0.2, 0.2],
        "test_top5": [0.9] * 6,
        "test_macro_f1": [0.3] * 6,
    })
    out = summarize(df, ["hands", "all"])
    assert list(out.index) == ["hands", "all"]
    assert out.loc["all", ("test_top1", "mean")] == pytest.approx(0.5)
    assert out.loc["all", ("test_top1", "std")] == pytest.approx(0.1)
    assert out.loc["hands", ("test_top1", "std")] == pytest.approx(0.0)
    assert out.loc["hands", ("runs", "")] == 3
