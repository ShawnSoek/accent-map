import json

import numpy as np
import pytest

from fakes import fake_config, tone
from server.model_api import load_model
from server.session import (
    CONFIDENT,
    LISTENING,
    NO_SPEECH,
    UNCERTAIN,
    AnalysisSession,
    SessionSettings,
    decide_status,
)

SR = 16000


def make_session(n_classes=4, **settings):
    model = load_model(fake_config(n_classes))
    return AnalysisSession(model, SessionSettings(**{"hop_seconds": 1.0, **settings})), model


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_tone_goes_from_listening_to_confident():
    session, model = make_session()
    events = list(session.analyze_all(tone(model.class_frequency("c2"), 8.0)))
    assert [e.t for e in events] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert events[0].status == LISTENING            # only 1 s heard so far
    assert all(e.status == CONFIDENT for e in events[1:])
    assert events[-1].top[0]["id"] == "c2"
    assert list(events[-1].raw) == list(events[-1].smoothed) == model.class_ids


def test_silence_gives_no_speech_and_no_prediction():
    session, _ = make_session()
    events = list(session.analyze_all(silence(5.0)))
    assert {e.status for e in events} == {NO_SPEECH}
    assert all(e.raw is None and e.smoothed is None and e.top == [] and e.infer_ms is None for e in events)


def test_pause_holds_the_last_result():
    session, model = make_session()
    audio = np.concatenate([tone(model.class_frequency("c1"), 3.0), silence(6.0)])
    events = list(session.analyze_all(audio))
    last_result = [e for e in events if e.raw is not None][-1]
    paused = [e for e in events if e.status == NO_SPEECH]
    assert [e.t for e in paused] == [8.0, 9.0]       # windows with < 0.75 s of sound
    assert all(e.smoothed == last_result.smoothed for e in paused)


def test_live_chunks_and_whole_file_give_identical_events():
    model = load_model(fake_config(5))
    audio = np.concatenate([tone(model.class_frequency("c1"), 3.0), silence(1.0),
                            tone(model.class_frequency("c3"), 4.3)])
    file_events = list(AnalysisSession(model, SessionSettings(hop_seconds=0.5)).analyze_all(audio))

    live = AnalysisSession(model, SessionSettings(hop_seconds=0.5))
    live_events, rng, pos = [], np.random.default_rng(0), 0
    while pos < len(audio):
        n = int(rng.integers(50, 3000))       # irregular chunk sizes, like a browser
        live_events += [live.analyze(p) for p in live.push(audio[pos:pos + n])]
        pos += n
    live_events += [live.analyze(p) for p in live.flush()]

    strip = [{k: v for k, v in e.to_dict().items() if k != "infer_ms"} for e in file_events]
    assert strip == [{k: v for k, v in e.to_dict().items() if k != "infer_ms"} for e in live_events]
    assert file_events[-1].t == pytest.approx(len(audio) / SR)   # flush covered the tail


def test_skipped_windows_still_count_as_heard_speech():
    session, model = make_session()
    pending = []
    for _ in range(6):                                  # 6 s in 1 s chunks, model "busy"
        pending += session.push(tone(model.class_frequency("c0"), 1.0))
    event = session.analyze(pending[-1])                # only the newest window is analyzed
    assert event.speech_seconds == pytest.approx(6.0)
    assert event.status == CONFIDENT


def test_switching_accent_moves_the_top_class_once():
    session, model = make_session()
    audio = np.concatenate([tone(model.class_frequency("c0"), 6.0), tone(model.class_frequency("c3"), 10.0)])
    tops = [e.top[0]["id"] for e in session.analyze_all(audio)]
    changes = [(a, b) for a, b in zip(tops, tops[1:]) if a != b]
    assert tops[0] == "c0" and tops[-1] == "c3"
    assert len(changes) == 1                             # no flicker back and forth


@pytest.mark.parametrize("n_classes", [2, 20])
def test_any_class_count(n_classes):
    session, model = make_session(n_classes)
    target = model.class_ids[-1]
    events = list(session.analyze_all(tone(model.class_frequency(target), 4.0)))
    assert events[-1].top[0]["id"] == target
    assert len(events[-1].top) == min(3, n_classes)
    assert len(events[-1].smoothed) == n_classes


def test_events_are_json_serializable():
    session, model = make_session()
    for event in session.analyze_all(tone(model.class_frequency("c1"), 3.0)):
        json.dumps(event.to_dict())


def test_decide_status_thresholds():
    s = SessionSettings()
    assert decide_status(np.array([0.9, 0.1]), speech_total=1.9, s=s) == LISTENING
    assert decide_status(np.array([0.6, 0.3, 0.1]), speech_total=5, s=s) == CONFIDENT
    assert decide_status(np.array([0.55, 0.45]), speech_total=5, s=s) == UNCERTAIN    # margin too small
    assert decide_status(np.array([0.45, 0.2, 0.2, 0.15]), speech_total=5, s=s) == UNCERTAIN  # top too low


def test_settings_keep_the_model_contract():
    with pytest.raises(ValueError, match="min_audio_seconds"):
        SessionSettings(min_audio_seconds=0.2)
