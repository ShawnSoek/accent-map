import numpy as np
import pytest

from server.windowing import Windower

SR = 16000


def ramp(seconds):
    """Each sample's value is its position, so window contents can be checked exactly."""
    return np.arange(int(round(seconds * SR)), dtype=np.float32)


def push_in_chunks(windower, x, chunk):
    out = []
    for i in range(0, len(x), chunk):
        out += windower.push(x[i:i + chunk])
    return out


def test_windows_end_on_hop_boundaries():
    windows = push_in_chunks(Windower(SR, 5.0, 0.5, 1.0), ramp(3.3), 1234)
    assert [w.t for w in windows] == [1.0, 1.5, 2.0, 2.5, 3.0]
    assert [w.index for w in windows] == [0, 1, 2, 3, 4]


def test_one_second_hop():
    windows = Windower(SR, 5.0, 1.0, 1.0).push(ramp(6.5))
    assert [w.t for w in windows] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize("chunk", [37, 1000, 4096, 10**9])
def test_chunk_size_does_not_change_windows(chunk):
    x = ramp(12.3)
    expected = Windower(SR, 5.0, 0.5, 1.0).push(x)
    got = push_in_chunks(Windower(SR, 5.0, 0.5, 1.0), x, chunk)
    assert [w.end_sample for w in got] == [w.end_sample for w in expected]
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a.audio, b.audio)
        np.testing.assert_array_equal(a.new_audio, b.new_audio)


def test_window_grows_then_slides_through_ring_buffer():
    x = ramp(12.3)
    windows = push_in_chunks(Windower(SR, 5.0, 0.5, 1.0), x, 999)
    for w in windows:
        np.testing.assert_array_equal(w.audio, x[max(0, w.end_sample - 5 * SR):w.end_sample])
    lengths = [len(w.audio) / SR for w in windows]
    assert lengths[:3] == [1.0, 1.5, 2.0] and lengths[-1] == 5.0


def test_new_audio_is_everything_since_previous_window():
    x = ramp(4.0)
    previous_end = 0
    for w in push_in_chunks(Windower(SR, 5.0, 1.0, 1.0), x, 3000):
        np.testing.assert_array_equal(w.new_audio, x[previous_end:w.end_sample])
        previous_end = w.end_sample


def test_first_window_waits_for_min_seconds():
    windows = Windower(SR, 5.0, 0.5, 1.2).push(ramp(2.0))
    assert [w.t for w in windows] == [1.5, 2.0]
    assert len(windows[0].new_audio) == 1.5 * SR


def test_flush_covers_trailing_audio():
    x = ramp(6.62)
    windower = Windower(SR, 5.0, 1.0, 1.0)
    assert windower.push(x)[-1].t == 6.0
    last = windower.flush()
    assert last.end_sample == len(x) and last.t == pytest.approx(6.62)
    np.testing.assert_array_equal(last.audio, x[-5 * SR:])
    np.testing.assert_array_equal(last.new_audio, x[6 * SR:])
    assert windower.flush() is None


def test_flush_returns_nothing_when_nothing_is_left():
    exact = Windower(SR, 5.0, 1.0, 1.0)
    exact.push(ramp(3.0))
    assert exact.flush() is None
    too_short = Windower(SR, 5.0, 1.0, 1.0)
    assert too_short.push(ramp(0.7)) == []
    assert too_short.flush() is None


@pytest.mark.parametrize("window, hop, min_s", [
    (5.0, 6.0, 1.0),    # hop longer than window
    (5.0, 0.0, 1.0),
    (5.0, 0.5, 6.0),    # min longer than window
    (1.0, 0.75, 0.9),   # first boundary (1.5 s) after the window length
])
def test_rejects_bad_settings(window, hop, min_s):
    with pytest.raises(ValueError):
        Windower(SR, window, hop, min_s)


def test_rejects_multichannel_chunk():
    with pytest.raises(ValueError, match="1-D"):
        Windower(SR, 5.0, 0.5, 1.0).push(np.zeros((100, 2), np.float32))
