"""File mode can start anywhere, and still cuts exactly the windows the live path would."""

import numpy as np
import pytest

from fakes import fake_config, tone
from server.file_analysis import FileAnalysis
from server.model_api import load_model
from server.session import CONFIDENT, AnalysisSession, SessionSettings, WindowAnalyzer

SR = 16000
SETTINGS = SessionSettings(window_seconds=10.0, hop_seconds=2.0)   # min_audio 1 s -> windows at 2, 4, ...


@pytest.fixture
def model():
    return load_model(fake_config(4))


@pytest.fixture
def audio(model):
    """23.3 s: class c0 for 7 s, 2 s silence, class c2 for 14.3 s."""
    return np.concatenate([tone(model.class_frequency("c0"), 7.0), np.zeros(2 * SR, np.float32),
                           tone(model.class_frequency("c2"), 14.3)])


def test_windows_and_speech_totals_match_the_live_path(model, audio):
    session = AnalysisSession(model, SETTINGS)
    live = session.push(audio) + session.flush()
    from_file = list(FileAnalysis(model, SETTINGS, audio).pendings_from(0))
    assert [p.window.t for p in from_file] == [p.window.t for p in live]
    for a, b in zip(from_file, live):
        assert (a.window.index, a.window.end_sample, a.speech_total) == (b.window.index, b.window.end_sample,
                                                                        b.speech_total)
        np.testing.assert_array_equal(a.window.audio, b.window.audio)
        np.testing.assert_array_equal(a.window.new_audio, b.window.new_audio)


def test_from_the_start_gives_the_same_events_as_a_live_session(model, audio):
    analyzer = WindowAnalyzer(model, SETTINGS)
    from_file = [analyzer.analyze(p).to_dict() for p in FileAnalysis(model, SETTINGS, audio).pendings_from(0)]
    live = [e.to_dict() for e in AnalysisSession(model, SETTINGS).analyze_all(audio)]
    strip = lambda d: {k: v for k, v in d.items() if k != "infer_ms"}  # noqa: E731
    assert [strip(d) for d in from_file] == [strip(d) for d in live]


@pytest.mark.parametrize("start, first_window_end", [
    (0.0, 2.0),      # before the first window: start with it
    (1.5, 2.0),
    (2.0, 2.0),      # exactly on a boundary
    (7.9, 6.0),      # the window ending at or just before the position
    (23.3, 23.3),    # the final partial window
    (100.0, 23.3),
])
def test_start_position(model, audio, start, first_window_end):
    analysis = FileAnalysis(model, SETTINGS, audio)
    assert next(analysis.pendings_from(start)).window.t == pytest.approx(first_window_end)


def test_after_a_jump_the_window_is_full_and_there_is_no_warmup(model, audio):
    analysis = FileAnalysis(model, SETTINGS, audio)
    first = next(analysis.pendings_from(15.0))
    assert first.window.t == 14.0 and len(first.window.audio) == 10 * SR
    assert first.speech_total == pytest.approx(12.0)      # 7 s + 5 s of tone before 14 s, not reset
    event = WindowAnalyzer(model, SETTINGS).analyze(first)
    assert event.status == CONFIDENT and event.top[0]["id"] == "c2"
    assert event.smoothed == pytest.approx(event.raw, abs=1e-12)   # fresh smoothing after a jump


def test_duration_and_window_times(model, audio):
    analysis = FileAnalysis(model, SETTINGS, audio)
    assert analysis.duration == pytest.approx(23.3)
    times = analysis.window_times
    assert times[:3] == [2.0, 4.0, 6.0] and times[-2:] == [22.0, pytest.approx(23.3)]
