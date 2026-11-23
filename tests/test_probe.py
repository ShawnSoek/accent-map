"""Reading the verified features, the speaker-balanced measures, and the per-layer probe that
finds where the accent is (synthetic features: the signal sits in one layer only)."""

import json

import numpy as np
import pytest

from fakes import make_accent_data
from training.feature_store import NotVerifiedError, open_set, read_labels
from training.metrics import (balanced_accuracy, bootstrap_interval, calibration_error, confusion, report,
                              speaker_accuracies)
from training.probe import class_weights, layer_stats, probe_probabilities, train_probes

LAYERS, DIM, CLASSES = 4, 8, 3
SCALE = np.array([1, 1, 1, 4], dtype=np.float32)       # layer 3 stored divided by 4, as layers 22-23 are
STORAGE = {3: 4}


@pytest.fixture
def folder(tmp_path):
    features, table, sets = make_accent_data(tmp_path)
    return features, read_labels(table, [f"k{c}" for c in range(CLASSES)]), sets


def test_sets_read_with_the_scaling_undone(folder):
    features, info, sets = folder
    train = open_set(features, "train", STORAGE)
    assert len(train) == 360 and train.shape == (LAYERS, DIM)
    assert train.ids[:2] == [sets["train"][0][0], sets["train"][0][2]]          # part 0 first: items 0, 2, ...
    ids, x = train.read(1, 0, 3)
    want = np.asarray(sets["train"][1][1::2][:3], dtype=np.float16).astype(np.float32) * SCALE[:, None]
    assert ids == sets["train"][0][1::2][:3] and np.array_equal(x, want)
    seen = [i for ids, _ in train.blocks(50, np.random.default_rng(0)) for i in ids]
    assert sorted(seen) == sorted(train.ids) and seen != train.ids                 # all once, shuffled blocks
    assert info[train.ids[0]].label == 0 and info[train.ids[0]].speaker.startswith("train-c0")


def test_test_sets_stay_closed_and_unverified_folders_too(folder, tmp_path):
    features, _, _ = folder
    with pytest.raises(PermissionError, match="final evaluation"):
        open_set(features, "test", STORAGE)
    assert len(open_set(features, "test", STORAGE, allow_test=True)) == 12
    (features / "verified.json").unlink()
    with pytest.raises(NotVerifiedError):
        open_set(features, "train", STORAGE)


def test_unknown_labels_are_refused(tmp_path):
    table = tmp_path / "t.tsv"
    table.write_text("id\tset\taugment\tclips\tlabel\tspeaker\tgender\tseconds\ntrain/0\ttrain\t0\tx\tzz\ts\tf\t1\n")
    with pytest.raises(ValueError, match="not in the class list"):
        read_labels(table, ["k0"])


def test_balanced_accuracy_counts_speakers_and_classes_equally():
    # class 0: speaker a (4 clips, all right), speaker b (1 clip, wrong); class 1: one speaker, half right
    y = np.array([0, 0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0, 1, 1, 0])
    speakers = ["a", "a", "a", "a", "b", "c", "c"]
    assert [list(a) for a in speaker_accuracies(y, pred, speakers, 2)] == [[1.0, 0.0], [0.5]]
    assert balanced_accuracy(y, pred, speakers, 2) == pytest.approx((0.5 + 0.5) / 2)
    assert (pred == y).mean() == pytest.approx(5 / 7)                             # what clip counting would say
    assert confusion(y, pred, 2).tolist() == [[4, 1], [1, 1]]
    low, high = bootstrap_interval(y, pred, speakers, 2, rounds=500)
    assert 0.25 <= low <= 0.5 <= high <= 0.75
    assert bootstrap_interval(y, y, speakers, 2) == (1.0, 1.0)


def test_calibration_and_report():
    y = np.array([0, 1, 1, 0])
    sure = np.eye(2)[y]
    assert calibration_error(sure, y) == 0.0
    half = np.full((4, 2), 0.5)
    half[:, 0] += 1e-6                                                             # says 50 %, right half the time
    assert calibration_error(half, y) == pytest.approx(0.0, abs=1e-5)
    out = report(sure, y, ["a", "b", "b", "c"], ["f", "m", "m", "other"], ["x", "y"])
    assert out["balanced_accuracy"] == 1.0 and out["interval_95"] == (1.0, 1.0)
    assert out["per_class"]["y"] == {"speakers": 1, "items": 2, "accuracy": 1.0}
    assert set(out["by_gender"]) == {"f", "m", "other"} and out["log_loss"] == pytest.approx(0.0, abs=1e-9)
    json.dumps(out)                                                                # can be saved as is


def test_layer_stats_match_numpy(folder):
    features, _, sets = folder
    train = open_set(features, "train", STORAGE)
    mean, std = layer_stats(train, block=37)
    rows = np.concatenate([train.read(*s)[1] for s in train.spans(1000)])
    assert np.allclose(mean, rows.mean(0), atol=1e-4) and np.allclose(std, rows.std(0), atol=1e-4)
    assert mean[3].mean() == pytest.approx(10, abs=0.5)                            # scaling undone


def test_class_weights_make_classes_equal():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w.tolist() == pytest.approx([4 / 6, 2.0]) and (w * [3, 1]).sum() == pytest.approx(4)
    with pytest.raises(ValueError, match="without training items"):
        class_weights(np.array([0, 0]), 2)


def test_probe_finds_the_layer_with_the_accent(folder):
    features, info, _ = folder
    train, dev = open_set(features, "train", STORAGE), open_set(features, "dev", STORAGE)
    mean, std = layer_stats(train)
    model = train_probes(train, info, mean, std, CLASSES, epochs=15, batch=32, lr=1e-2, block=100,
                         log=lambda m: None)
    ids, probs = probe_probabilities(model, dev, block=40)
    assert ids == dev.ids and probs.shape == (len(dev), LAYERS, CLASSES)
    y = np.array([info[i].label for i in ids])
    speakers = [info[i].speaker for i in ids]
    acc = [balanced_accuracy(y, probs[:, k].argmax(1), speakers, CLASSES) for k in range(LAYERS)]
    assert acc[2] > 0.9 and max(acc[:2] + acc[3:]) < 0.6
