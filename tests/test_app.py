"""Server tests with the fake model: /api/model, file upload (NDJSON), live WebSocket."""

import io
import json

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from fakes import fake_config, tone
from server.app import Engine, build_engine, create_app
from server.audio import decode_file
from server.model_api import load_model
from server.session import AnalysisSession, SessionSettings

SR = 16000


def make_client(n_classes=4, hop=0.5, **backend_options):
    model = load_model(fake_config(n_classes, **backend_options))
    engine = Engine(model=model, settings=SessionSettings(hop_seconds=hop))
    return TestClient(create_app(engine=engine)), model


def wav_bytes(audio, rate):
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def ndjson(response):
    return [json.loads(line) for line in response.text.splitlines()]


def live_run(ws, audio, rate, chunk=4800):
    """Send audio as fast as possible, then stop; return all server messages."""
    ws.send_json({"type": "start", "sample_rate": rate})
    messages = [ws.receive_json()]
    for i in range(0, len(audio), chunk):
        ws.send_bytes(audio[i:i + chunk].astype("<f4").tobytes())
    ws.send_json({"type": "stop"})
    while messages[-1]["type"] not in ("done", "error"):
        messages.append(ws.receive_json())
    return messages


# --- /api/model ----------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [3, 7])
def test_model_info_follows_the_config(n_classes):
    client, model = make_client(n_classes)
    info = client.get("/api/model").json()
    assert [c["id"] for c in info["classes"]] == model.class_ids
    assert set(info["classes"][0]) == {"id", "label", "description", "zone"}   # no model_label
    assert info["audio"]["update_seconds"] == 0.5
    assert info["audio"]["window_seconds"] == 5.0


# --- POST /api/analyze ---------------------------------------------------------------

def test_upload_streams_events_in_time_order():
    client, model = make_client()
    audio = tone(model.class_frequency("c2"), 6.3, sample_rate=44100)
    lines = ndjson(client.post("/api/analyze", files={"file": ("a.wav", wav_bytes(audio, 44100), "audio/wav")}))
    assert lines[0]["type"] == "meta" and lines[0]["duration"] == pytest.approx(6.3, abs=1e-3)
    events = [m for m in lines if m["type"] == "event"]
    assert lines[-1] == {"type": "done", "events": len(events)}
    assert [e["t"] for e in events][:3] == [1.0, 1.5, 2.0]
    assert events[-1]["t"] == pytest.approx(6.3, abs=1e-3)      # final partial window
    assert events[-1]["top"][0]["id"] == "c2"


def test_upload_uses_the_same_logic_as_a_session():
    client, model = make_client()
    data = wav_bytes(np.concatenate([tone(model.class_frequency("c0"), 3.0), tone(model.class_frequency("c3"), 4.0)]),
                     SR)
    served = [m for m in ndjson(client.post("/api/analyze", files={"file": ("a.wav", data)})) if m["type"] == "event"]
    direct = AnalysisSession(model, SessionSettings(hop_seconds=0.5)).analyze_all(decode_file(io.BytesIO(data), SR))
    strip = lambda d: {k: v for k, v in d.items() if k not in ("infer_ms", "type")}  # noqa: E731
    assert [strip(m) for m in served] == [strip(e.to_dict()) for e in direct]


@pytest.mark.parametrize("content, status, message", [
    (b"", 400, "empty"),
    (b"this is not audio" * 100, 415, "Supported: WAV, FLAC, OGG, MP3"),
    (wav_bytes(tone(300, 0.3), SR), 422, "too short"),
])
def test_upload_rejects_bad_files(content, status, message):
    client, _ = make_client()
    r = client.post("/api/analyze", files={"file": ("x.bin", content)})
    assert r.status_code == status
    assert message in r.json()["detail"]
    assert "BytesIO" not in r.json()["detail"]   # no internals in user-facing messages


# --- WS /ws/live ---------------------------------------------------------------------

def test_live_stream_at_48k():
    client, model = make_client()
    audio = tone(model.class_frequency("c1"), 4.0, sample_rate=48000)
    with client.websocket_connect("/ws/live") as ws:
        messages = live_run(ws, audio, 48000)
    assert messages[0] == {"type": "ready", "sample_rate": 48000, "update_seconds": 0.5}
    events = [m for m in messages if m["type"] == "event"]
    done = messages[-1]
    assert done["type"] == "done" and done["events"] == len(events)
    ts = [e["t"] for e in events]
    assert ts == sorted(ts) and ts[-1] == pytest.approx(4.0, abs=1e-3)
    assert len(events) + done["dropped"] == 7                    # windows at 1.0, 1.5, ..., 4.0
    assert events[-1]["top"][0]["id"] == "c1"
    assert all(e["latency_ms"] >= e["queue_ms"] >= 0 for e in events)


def test_live_skips_stale_windows_when_the_model_is_slow():
    client, model = make_client(delay_seconds=0.2)               # slower than audio arrives here
    audio = tone(model.class_frequency("c3"), 6.0)
    with client.websocket_connect("/ws/live") as ws:
        messages = live_run(ws, audio, SR, chunk=1600)
    events = [m for m in messages if m["type"] == "event"]
    assert messages[-1]["dropped"] > 0
    assert len(events) + messages[-1]["dropped"] == 11           # every window analyzed or skipped
    assert events[-1]["t"] == 6.0                                # but the newest audio always is
    assert events[-1]["speech_seconds"] == pytest.approx(6.0)    # skipped audio still counts


@pytest.mark.parametrize("first, error", [
    (b"\x00\x00\x80\x3f", "first message must be JSON"),
    ({"type": "start"}, "sample_rate"),
    ({"type": "start", "sample_rate": 1000}, "between 8000 and 192000"),
    ({"type": "go", "sample_rate": 16000}, "first message must be"),
])
def test_live_rejects_bad_start(first, error):
    client, _ = make_client()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(first) if isinstance(first, bytes) else ws.send_json(first)
        msg = ws.receive_json()
    assert msg["type"] == "error" and error in msg["message"]


@pytest.mark.parametrize("payload, error", [
    (b"\x00\x00\x80", "divisible by 4"),
    (np.full(100, np.nan, "<f4").tobytes(), "NaN"),
    ('{"type": "pause"}', "expected"),
])
def test_live_rejects_bad_messages(payload, error):
    client, _ = make_client()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "sample_rate": SR})
        ws.receive_json()
        ws.send_bytes(payload) if isinstance(payload, bytes) else ws.send_text(payload)
        msg = ws.receive_json()
    assert msg["type"] == "error" and error in msg["message"]


# --- Startup: load + pick the update interval ---------------------------------------

def test_build_engine_measures_and_picks_interval(tmp_path):
    config = fake_config(3, delay_seconds=0.05).model_dump(exclude={"model_dir"})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    engine = build_engine(tmp_path)
    assert 40 <= engine.measured_infer_ms <= 300
    assert engine.settings.hop_seconds == 0.5                    # 50 ms easily fits 0.5 s
    slow = fake_config(3, delay_seconds=0.7).model_dump(exclude={"model_dir"})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(slow))
    assert build_engine(tmp_path).settings.hop_seconds == 1.0    # 0.7 s * 1.2 -> 1.0 s
    assert build_engine(tmp_path, update_seconds=2.0).settings.hop_seconds == 2.0
