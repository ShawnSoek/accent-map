"""Speaker-disjoint splits: no speaker or held-out sentence leaks into training, caps hold, the same every run."""

import random
from pathlib import Path

import pytest

from training.cv_split import DatasetSpec, check_disjoint, counts, load_spec, make_splits

REPO = Path(__file__).resolve().parents[1]
A, B = "Accent A", "Accent B"


def spec(**changes):
    base = {
        "release": "test", "archive_sha256": "0" * 64, "source": "commonvoice", "max_clips_per_speaker": 6,
        "split": {"seed": "s", "eligible_min_clips": 5, "test": {"fraction": 0.2, "min_speakers": 2},
                  "dev": {"fraction": 0.1, "min_speakers": 1}},
        "classes": [{"id": "a", "accent": A}, {"id": "b", "accent": B}],
    }
    base.update(changes)
    return DatasetSpec.model_validate(base)


def make_rows(seed=0):
    """Accent A: 20 speakers, B: 10. Clip counts 2..11. Sentences shared across speakers."""
    rng = random.Random(seed)
    rows = []
    for accent, n in ((A, 20), (B, 10)):
        for k in range(n):
            for j in range(2 + (k % 10)):
                rows.append({"client_id": f"{accent[-1]}{k}", "path": f"{accent[-1]}{k}_{j}.mp3",
                             "sentence_id": f"sent{rng.randrange(40)}", "accents": accent,
                             "gender": "female" if k % 3 == 0 else "male"})
    rows.append({"client_id": "mixed", "path": "m.mp3", "sentence_id": "x", "accents": f"{A}|{B}", "gender": ""})
    rows.append({"client_id": "other", "path": "o.mp3", "sentence_id": "y", "accents": "Somewhere", "gender": ""})
    return rows


def durations(rows):
    return {r["path"]: 5.0 for r in rows}


def test_splits_are_disjoint_capped_and_complete():
    rows = make_rows()
    result = make_splits(rows, spec(), durations(rows))
    check_disjoint(result, sentences_too=True)
    speakers = {name: {c.speaker for c in clips} for name, clips in result.clips.items()}
    # A: 14 speakers with >= 5 clips -> test max(2, round(2.8)) = 3, dev max(1, round(1.4)) = 1.
    assert len({s for s in speakers["test"] if s.startswith("A")}) == 3
    assert len({s for s in speakers["dev"] if s.startswith("A")}) == 1
    # B: 7 eligible -> test 2 (minimum), dev 1.
    assert len({s for s in speakers["test"] if s.startswith("B")}) == 2
    all_clips = [c for clips in result.clips.values() for c in clips]
    assert "mixed" not in {c.speaker for c in all_clips} and "other" not in {c.speaker for c in all_clips}
    per_speaker = {}
    for c in all_clips:
        per_speaker[c.speaker] = per_speaker.get(c.speaker, 0) + 1
    assert max(per_speaker.values()) <= 6
    held = [c for name in ("dev", "test") for c in result.clips[name]]
    assert all(per_speaker[c.speaker] >= 5 for c in held)              # only eligible speakers held out
    assert {c.label for c in all_clips} == {"a", "b"}
    # Every other one-accent speaker trains, unless all their clips read held-out sentences.
    held_speakers = speakers["dev"] | speakers["test"]
    held_sentences = {c.sentence_id for c in held}
    expected = {r["client_id"] for r in rows if r["client_id"] not in held_speakers | {"mixed", "other"}
                and r["sentence_id"] not in held_sentences}
    assert speakers["train"] == expected and len(expected) > 0


def test_training_clips_with_held_out_sentences_are_dropped():
    rows = make_rows()
    with_drop = make_splits(rows, spec(), durations(rows))
    split = spec().split.model_copy(update={"drop_train_clips_with_dev_test_sentences": False})
    without = make_splits(rows, spec(split=split.model_dump()), durations(rows))
    assert with_drop.dropped_for_sentences > 0 and without.dropped_for_sentences == 0
    with pytest.raises(AssertionError, match="sentences"):
        check_disjoint(without, sentences_too=True)
    check_disjoint(without, sentences_too=False)


def test_the_split_does_not_depend_on_row_order():
    rows = make_rows()
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    first = make_splits(rows, spec(), durations(rows))
    second = make_splits(shuffled, spec(), durations(rows))
    for name in ("train", "dev", "test"):
        assert sorted(first.clips[name], key=lambda c: c.path) == sorted(second.clips[name], key=lambda c: c.path)
    other_seed = make_splits(rows, spec(split={**spec().split.model_dump(), "seed": "other"}), durations(rows))
    assert {c.speaker for c in other_seed.clips["test"]} != {c.speaker for c in first.clips["test"]}


def test_too_few_speakers_is_an_error():
    rows = make_rows()
    big = {**spec().split.model_dump(), "test": {"fraction": 0.2, "min_speakers": 7}}
    with pytest.raises(ValueError, match="b: 7 speakers .* need 1 \\+ 7"):
        make_splits(rows, spec(split=big), durations(rows))


def test_missing_duration_is_an_error():
    rows = make_rows()
    d = durations(rows)
    del d["A3_0.mp3"]
    with pytest.raises(KeyError, match="A3_0.mp3"):
        make_splits(rows, spec(), d)


def test_counts():
    rows = make_rows()
    result = make_splits(rows, spec(), durations(rows))
    table = counts(result, ["a", "b"])
    assert sum(table[c][s]["clips"] for c in table for s in table[c]) == sum(len(v) for v in result.clips.values())
    assert table["b"]["test"]["speakers"] == 2 and table["b"]["test"]["hours"] == pytest.approx(
        sum(c.seconds for c in result.clips["test"] if c.label == "b") / 3600)


def test_the_committed_spec_is_valid():
    s = load_spec(REPO / "training/cv27_es10/dataset.yaml")
    assert len(s.classes) == 10 and s.max_clips_per_speaker == 50
    with pytest.raises(ValueError, match="unique"):
        DatasetSpec.model_validate({**s.model_dump(), "classes": [s.classes[0].model_dump()] * 2})
