import pytest

from fakes import fake_config
from server.model_api import load_model
from server.pacing import choose_update_interval, measure_inference_ms


@pytest.mark.parametrize("infer_ms, expected", [
    (300, 0.5),     # fast hardware keeps the configured 0.5 s
    (416, 0.5),     # 416 ms * 1.2 = 499 ms still fits
    (420, 0.75),
    (805, 1.0),     # this laptop's CPU, 5 s window (step 4)
    (828, 1.0),     # its p90
    (840, 1.25),
    (2000, 2.5),
])
def test_choose_update_interval(infer_ms, expected):
    assert choose_update_interval(0.5, infer_ms) == expected


def test_never_faster_than_the_config_asks():
    assert choose_update_interval(1.0, 10) == 1.0


def test_measure_uses_full_windows_and_times_them():
    model = load_model(fake_config(3, delay_seconds=0.03))
    lengths = []
    predict = model.predict
    model.predict = lambda audio: lengths.append(len(audio)) or predict(audio)
    ms = measure_inference_ms(model, window_seconds=5.0, runs=3, warmup=1)
    assert lengths == [5 * 16000] * 4
    assert 25 <= ms <= 200
