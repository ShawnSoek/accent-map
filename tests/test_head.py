"""The classifier's inputs (chosen layers, standardized and averaged, written once) and its
training with early stopping on dev, on synthetic features whose accent sits in layer 2."""

import numpy as np
import pytest

from fakes import make_accent_data
from training.feature_store import open_set, read_labels
from training.head import (AccentHead, HeadConfig, build_inputs, column_stats, combine, layer_choices, load_inputs,
                           predict, train_head)
from training.metrics import balanced_accuracy
from training.probe import layer_stats

CLASSES = 3
STORAGE = {3: 4}


@pytest.fixture
def data(tmp_path):
    features, table, _ = make_accent_data(tmp_path)
    info = read_labels(table, [f"k{c}" for c in range(CLASSES)])
    train, dev = open_set(features, "train", STORAGE), open_set(features, "dev", STORAGE)
    mean, std = layer_stats(train)
    choices = {"signal": [2], "around": [1, 2, 3], "none": [0, 1]}
    for fset in (train, dev):
        build_inputs(fset, mean, std, choices, tmp_path / "inputs", block=50)
    return tmp_path / "inputs", info, train, mean, std


def arrays(folder, set_name, info):
    ids, x = load_inputs(folder, set_name)
    return ids, x, np.array([info[i].label for i in ids]), [info[i].speaker for i in ids]


def test_combine_standardizes_then_averages():
    x = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    mean, std = np.ones((3, 2), np.float32), np.full((3, 2), 2, np.float32)
    want = (((x[:, 0] - 1) / 2) + ((x[:, 2] - 1) / 2)) / 2
    assert np.allclose(combine(x, mean, std, [0, 2]), want)


def test_inputs_are_written_in_the_sets_order(data):
    folder, _, train, mean, std = data
    ids, x = load_inputs(folder / "around", "train")
    assert ids == train.ids and x.dtype == np.float16 and x.shape == (len(train), 8)
    _, rows = train.read(*train.spans(10 ** 6)[0])                                   # part 0, in order
    assert np.allclose(x[:len(rows)], combine(rows, mean, std, [1, 2, 3]), atol=2e-3)


def test_column_stats_match_numpy():
    x = np.random.default_rng(0).normal(3, 2, size=(1000, 5)).astype(np.float16)
    mean, std = column_stats(x, chunk=97)
    assert np.allclose(mean, x.astype(np.float64).mean(0), atol=1e-4)
    assert np.allclose(std, x.astype(np.float64).std(0), atol=1e-4)


@pytest.mark.parametrize("hidden, dropout", [(0, 0.0), (16, 0.3)])
def test_head_learns_from_the_right_layer_only(data, hidden, dropout):
    folder, info, *_ = data
    config = HeadConfig(hidden=hidden, dropout=dropout, lr=1e-2, batch=32, max_epochs=12, patience=4)
    scores = {}
    for choice in ("signal", "none"):
        _, x, y, _ = arrays(folder / choice, "train", info)
        _, dx, dy, dspk = arrays(folder / choice, "dev", info)
        model, history = train_head(x, y, dx, dy, dspk, CLASSES, config, log=lambda m: None)
        again = balanced_accuracy(dy, predict(model, dx).argmax(1), dspk, CLASSES)
        assert again == pytest.approx(history["best_dev"])                            # the best epoch is kept
        assert history["best_dev"] == max(history["dev_history"]) and len(history["dev_history"]) <= 12
        scores[choice] = history["best_dev"]
    assert scores["signal"] > 0.9 and scores["none"] < 0.7


def test_predict_gives_probabilities():
    model = AccentHead(4, 3, hidden=5)
    p = predict(model, np.random.default_rng(1).normal(size=(10, 4)).astype(np.float16), batch=3)
    assert p.shape == (10, 3) and np.allclose(p.sum(1), 1, atol=1e-6)
    assert sum(t.numel() for t in AccentHead(4, 3).parameters()) == 4 * 3 + 3


def test_layer_choices_follow_the_probe():
    scores = [0.2, 0.3, 0.42, 0.44, 0.40, 0.46, 0.35, 0.13]          # best layer 5, near it 2, 3, 5
    assert layer_choices(scores) == {                                 # no 9-layer window: only 8 layers
        "layer 5": [5], "layers 4-6": [4, 5, 6], "layers 3-7": [3, 4, 5, 6, 7],
        "within 5% of best (3 layers)": [2, 3, 5], "all layers": list(range(8))}
    edge = layer_choices([0.1] * 24 + [0.5])                          # best at the end: windows shift inside
    assert edge["layers 22-24"] == [22, 23, 24] and edge["layers 16-24"] == list(range(16, 25))


def test_saved_head_gives_the_same_answers(data, tmp_path):
    from training.head import TrainedHead, save_head
    folder, info, train, mean, std = data
    _, x, y, _ = arrays(folder / "around", "train", info)
    _, dx, dy, dspk = arrays(folder / "around", "dev", info)
    config = HeadConfig(hidden=8, dropout=0.1, lr=1e-2, batch=32, max_epochs=3)
    model, _ = train_head(x, y, dx, dy, dspk, CLASSES, config, log=lambda m: None)
    save_head(tmp_path / "head.pt", model, ["k0", "k1", "k2"], [1, 2, 3], mean, std, config, temperature=2.0)
    head = TrainedHead(tmp_path / "head.pt")
    assert head.class_ids == ["k0", "k1", "k2"] and head.layers == [1, 2, 3] and head.temperature == 2.0
    _, rows = train.read(0, 0, 20)                                         # raw features, scaling undone
    direct = predict(model, combine(rows, mean, std, [1, 2, 3]).astype(np.float32))
    assert np.allclose(head.probabilities(rows, temperature=1.0), direct, atol=1e-5)
    warm = head.probabilities(rows)
    assert np.allclose(warm.sum(1), 1, atol=1e-6) and (warm.max(1) <= direct.max(1) + 1e-6).all()


def test_temperature_fixes_over_and_under_confidence():
    from training.head import fit_temperature
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 4)) * 2
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in true_p])
    softmax = lambda z: np.exp(z) / np.exp(z).sum(1, keepdims=True)        # noqa: E731
    assert fit_temperature(true_p, y) == pytest.approx(1.0, abs=0.1)
    assert fit_temperature(softmax(logits * 3), y) == pytest.approx(3.0, rel=0.1)       # too sure: cool down
    assert fit_temperature(softmax(logits / 2), y) == pytest.approx(0.5, rel=0.1)       # too unsure


def test_exported_head_matches_and_strangers_are_refused(data, tmp_path):
    from safetensors.torch import save_file

    from training.head import TrainedHead, export_head, save_head
    folder, info, train, mean, std = data
    _, x, y, _ = arrays(folder / "signal", "train", info)
    _, dx, dy, dspk = arrays(folder / "signal", "dev", info)
    config = HeadConfig(hidden=0, lr=1e-2, batch=32, max_epochs=3)
    model, _ = train_head(x, y, dx, dy, dspk, CLASSES, config, log=lambda m: None)
    save_head(tmp_path / "head.pt", model, ["k0", "k1", "k2"], [2], mean, std, config, temperature=1.3)
    original = TrainedHead(tmp_path / "head.pt")
    export_head(original, tmp_path / "head.safetensors", {3: 4.0}, {"trained_on": "test"})
    copy = TrainedHead(tmp_path / "head.safetensors")
    _, rows = train.read(0, 0, 30)
    np.testing.assert_array_equal(copy.probabilities(rows), original.probabilities(rows))
    assert (copy.class_ids, copy.layers, copy.temperature) == (["k0", "k1", "k2"], [2], 1.3)
    assert copy.saved["storage_scale"] == {3: 4.0} and copy.saved["info"] == {"trained_on": "test"}
    import torch
    save_file({"x": torch.zeros(2)}, str(tmp_path / "other.safetensors"), metadata={"format": "something else"})
    with pytest.raises(ValueError, match="not an accent-map classifier"):
        TrainedHead(tmp_path / "other.safetensors")
