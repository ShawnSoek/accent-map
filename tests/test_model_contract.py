"""The model interface contract (see server/model_api.py), for any backend and class count."""

import numpy as np
import pytest

from fakes import fake_config, tone
from server.audio import decode_file
from server.model_api import (
    MIN_AUDIO_SECONDS,
    AccentModel,
    ModelContractError,
    label_permutation,
    load_model,
)


def assert_contract(model: AccentModel, audio: np.ndarray) -> dict[str, float]:
    probs = model.predict(audio)
    assert list(probs) == model.class_ids  # exactly the config ids, in config order
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    return probs


# --- Fake backend: any class count ------------------------------------------------

@pytest.mark.parametrize("n_classes", [3, 6, 20])
def test_fake_model_contract(n_classes):
    model = load_model(fake_config(n_classes))
    rng = np.random.default_rng(0)
    for seconds in (MIN_AUDIO_SECONDS, 1.0, 5.0, 7.3):
        assert_contract(model, (0.1 * rng.standard_normal(int(seconds * 16000))).astype(np.float32))


@pytest.mark.parametrize("n_classes", [3, 20])
def test_fake_model_tone_selects_class(n_classes):
    model = load_model(fake_config(n_classes))
    for class_id in model.class_ids:
        probs = assert_contract(model, tone(model.class_frequency(class_id), 2.0))
        assert max(probs, key=probs.get) == class_id


def test_fake_model_silence_is_uniform():
    model = load_model(fake_config(4))
    probs = assert_contract(model, np.zeros(16000, dtype=np.float32))
    assert set(probs.values()) == {0.25}


@pytest.mark.parametrize("bad_audio, message", [
    (np.zeros((16000, 2), dtype=np.float32), "1-D"),
    (np.zeros(16000, dtype=np.int16), "floating point"),
    (np.zeros(int(MIN_AUDIO_SECONDS * 16000) - 1, dtype=np.float32), "too short"),
    (np.zeros(0, dtype=np.float32), "too short"),
    (np.full(16000, np.nan, dtype=np.float32), "NaN"),
])
def test_rejects_bad_audio(bad_audio, message):
    with pytest.raises(ValueError, match=message):
        load_model(fake_config(3)).predict(bad_audio)


class _Broken(AccentModel):
    def __init__(self, config, output):
        super().__init__(config)
        self.output = output

    def _predict(self, audio):
        return self.output


@pytest.mark.parametrize("output", [
    np.array([0.5, 0.5]),           # wrong class count
    np.array([0.5, 0.2, 0.2]),      # does not sum to 1
    np.array([1.2, -0.1, -0.1]),    # out of range
    np.array([np.nan, 0.5, 0.5]),
])
def test_contract_violations_are_caught(output):
    with pytest.raises(ModelContractError):
        _Broken(fake_config(3), output).predict(tone(300, 1.0))


def test_unknown_backend():
    config = fake_config(3).model_copy(update={"backend": "nope"})
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        load_model(config)


# --- Class order is read from the model, never assumed ------------------------------

def test_label_permutation_maps_by_label():
    config = fake_config(3)                       # model_labels LABEL 0, LABEL 1, LABEL 2
    model_order = ["LABEL 2", "LABEL 0", "LABEL 1"]
    perm = label_permutation(model_order, config)
    model_probs = np.array([0.7, 0.2, 0.1])       # i.e. LABEL 2 = 0.7
    in_config_order = model_probs[perm]
    assert dict(zip(config.class_ids, in_config_order)) == {"c0": 0.2, "c1": 0.1, "c2": 0.7}


@pytest.mark.parametrize("model_order, message", [
    (["LABEL 0", "LABEL 1"], "In config but not in model: \\['LABEL 2'\\]"),
    (["LABEL 0", "LABEL 1", "LABEL 2", "LABEL 3"], "In model but not in config: \\['LABEL 3'\\]"),
    (["LABEL 0", "LABEL 0", "LABEL 1"], "duplicate"),
])
def test_label_permutation_rejects_mismatch(model_order, message):
    with pytest.raises(ValueError, match=message):
        label_permutation(model_order, fake_config(3))


# --- The real stage-1 model (pytest -m model) ----------------------------------------

# Probabilities in config order, from scripts/check_model.py on 2026-11-13 with the
# B1+ head and soxr resampling. File names are hints, not verified labels.
EXPECTED = {
    "andino.wav":               [0.0201, 0.0312, 0.0143, 0.9004, 0.0161, 0.0179],
    "caribe-cuba-colombia.wav": [0.0212, 0.0292, 0.0406, 0.8725, 0.0190, 0.0174],
    "mexico.wav":               [0.0145, 0.9253, 0.0152, 0.0165, 0.0129, 0.0157],
    "spain.wav":                [0.9242, 0.0133, 0.0147, 0.0155, 0.0164, 0.0159],
}


@pytest.mark.model
def test_real_model_label_mapping(real_model):
    config_labels = [c.model_label for c in real_model.config.classes]
    assert sorted(real_model.model_labels) == sorted(config_labels)
    for config_pos, model_pos in enumerate(real_model._perm):
        assert real_model.model_labels[model_pos] == config_labels[config_pos]


@pytest.mark.model
def test_real_model_contract(real_model, sample_clips):
    audio = decode_file(sample_clips["andino.wav"], 16000)
    for seconds in (MIN_AUDIO_SECONDS, 1.0, 5.0, len(audio) / 16000):
        assert_contract(real_model, audio[: int(seconds * 16000)])


@pytest.mark.model
def test_real_model_sample_clip_regression(real_model, sample_clips):
    assert set(sample_clips) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        probs = assert_contract(real_model, decode_file(sample_clips[name], 16000))
        np.testing.assert_allclose(list(probs.values()), expected, atol=1e-3, err_msg=name)


@pytest.mark.model
def test_real_model_repeatable(real_model, sample_clips):
    audio = decode_file(sample_clips["spain.wav"], 16000)
    a, b = real_model.predict(audio), real_model.predict(audio)
    np.testing.assert_allclose(list(a.values()), list(b.values()), atol=1e-5)
