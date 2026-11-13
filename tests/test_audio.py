import numpy as np
import pytest
import soundfile as sf
import soxr

from fakes import tone
from server.audio import StreamingResampler, decode_file, resample, to_mono, voiced_seconds

SR = 16000


def peak_hz(x, sr):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spectrum)]


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(10), -0.5 * np.ones(10)], axis=1)
    np.testing.assert_allclose(to_mono(stereo), 0.25)
    mono = np.arange(5, dtype=np.float32)
    np.testing.assert_array_equal(to_mono(mono), mono)


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_resample_keeps_duration_and_pitch(rate):
    y = resample(tone(440, 2.0, sample_rate=rate), rate, SR)
    assert y.dtype == np.float32
    assert abs(len(y) - 2 * SR) <= 1
    assert abs(peak_hz(y, SR) - 440) < 1.0


def test_decode_stereo_44k_file_and_file_object(tmp_path):
    x = tone(440, 1.5, sample_rate=44100)
    path = tmp_path / "x.wav"
    sf.write(path, np.stack([x, x], axis=1), 44100, subtype="PCM_16")
    y = decode_file(path, SR)
    assert y.ndim == 1 and abs(len(y) - 1.5 * SR) <= 1
    assert abs(peak_hz(y, SR) - 440) < 1.0
    with open(path, "rb") as f:  # uploads arrive as file objects
        np.testing.assert_array_equal(decode_file(f, SR), y)


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("chunk", [128, 777, 4800])
def test_streaming_resampler_matches_whole_signal(rate, chunk):
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal(3 * rate)).astype(np.float32) + tone(300, 3.0, sample_rate=rate)
    whole = soxr.resample(x, rate, SR)
    r = StreamingResampler(rate, SR)
    streamed = np.concatenate([r.process(x[i:i + chunk]) for i in range(0, len(x), chunk)] + [r.flush()])
    assert len(streamed) == len(whole)
    np.testing.assert_allclose(streamed, whole, atol=1e-4)


def test_streaming_resampler_same_rate_is_passthrough():
    r = StreamingResampler(SR, SR)
    x = tone(300, 0.1)
    np.testing.assert_array_equal(r.process(x), x)
    assert len(r.flush()) == 0


def test_voiced_seconds():
    gate = -45.0
    assert voiced_seconds(np.zeros(SR, np.float32), SR, gate) == 0.0
    assert voiced_seconds(tone(300, 1.0, amplitude=0.3), SR, gate) == pytest.approx(1.0)    # about -13 dBFS
    assert voiced_seconds(tone(300, 1.0, amplitude=0.001), SR, gate) == 0.0                 # about -63 dBFS
    half = np.concatenate([np.zeros(SR // 2, np.float32), tone(300, 0.5, amplitude=0.3)])
    assert voiced_seconds(half, SR, gate) == pytest.approx(0.5)
    quiet = tone(300, 1.0, amplitude=0.014)  # about -40 dBFS
    assert voiced_seconds(quiet, SR, -45.0) == pytest.approx(1.0)
    assert voiced_seconds(quiet, SR, -35.0) == 0.0
