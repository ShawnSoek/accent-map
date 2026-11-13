"""HTTP and WebSocket server.

  GET  /api/model    model info for the page: classes, zones, caveats, update interval
  WS   /ws/live      live microphone audio in, events out (protocol below)
  POST /api/analyze  upload an audio file, events out as NDJSON (one JSON object per line)
  GET  /             the web page (web/), once it exists

Live protocol:
  client -> {"type": "start", "sample_rate": 48000}
  client -> binary messages: mono float32 little-endian samples at that rate
  client -> {"type": "stop"}            (analyze what is left, then finish)
  server -> {"type": "ready", ...} | {"type": "event", ...} | {"type": "done", ...}
            | {"type": "error", "message": ...} (then the server closes)

One model instance and one inference thread serve everyone; sessions take turns.
Live sessions analyze only the newest window when the model is busy (DECISIONS.md).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.audio import StreamingResampler, decode_file
from server.model_api import AccentModel, load_model
from server.pacing import choose_update_interval, measure_inference_ms
from server.paths import MODELS_DIR, REPO_ROOT
from server.session import AnalysisSession, Event, Pending, SessionSettings

log = logging.getLogger("uvicorn.error")

DEFAULT_MODEL_DIR = MODELS_DIR / "commonaccent_es6"
WEB_DIR = REPO_ROOT / "web"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_FILE_SECONDS = 15 * 60
SAMPLE_RATE_RANGE = (8000, 192000)
MAX_CHUNK_SECONDS = 2.0


@dataclass
class Engine:
    """The loaded model, the update interval chosen for this machine, and the inference thread."""
    model: AccentModel
    settings: SessionSettings
    measured_infer_ms: float | None = None
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference"))

    async def analyze(self, session: AnalysisSession, pending: Pending) -> Event:
        return await asyncio.get_running_loop().run_in_executor(self.executor, session.analyze, pending)


def build_engine(model_dir: str | Path, update_seconds: float | None = None) -> Engine:
    """Load the model and pick the update interval (measured unless given)."""
    model = load_model(model_dir)
    audio = model.config.audio
    measured = None
    if update_seconds is None:
        measured = measure_inference_ms(model, audio.window_seconds)
        update_seconds = choose_update_interval(audio.hop_seconds, measured)
    settings = SessionSettings(window_seconds=audio.window_seconds, hop_seconds=update_seconds)
    return Engine(model=model, settings=settings, measured_infer_ms=measured)


class ProtocolError(Exception):
    pass


class LiveStream:
    """Per-connection state: resampler, session, and the newest window waiting for the model."""

    def __init__(self, engine: Engine, sample_rate: int):
        self.engine = engine
        self.session = AnalysisSession(engine.model, engine.settings)
        self.resampler = StreamingResampler(sample_rate, engine.model.sample_rate)
        self.latest: tuple[Pending, float] | None = None
        self.dropped = 0
        self.closed = False
        self.wake = asyncio.Event()

    def feed(self, chunk: np.ndarray) -> None:
        self._offer(self.session.push(self.resampler.process(chunk)))

    def finish(self) -> None:
        self._offer(self.session.push(self.resampler.flush()) + self.session.flush())
        self.closed = True
        self.wake.set()

    def _offer(self, pendings: list[Pending]) -> None:
        if not pendings:
            return
        # Only the newest window is kept; any older one still waiting is skipped.
        self.dropped += len(pendings) - 1 + (self.latest is not None)
        self.latest = (pendings[-1], time.perf_counter())
        self.wake.set()

    async def events(self):
        """Yield (event, queue_ms, latency_ms) until finish() and nothing is left."""
        while True:
            if self.latest is None:
                if self.closed:
                    return
                await self.wake.wait()
                self.wake.clear()
                continue
            pending, ready = self.latest
            self.latest = None
            started = time.perf_counter()
            event = await self.engine.analyze(self.session, pending)
            yield event, (started - ready) * 1000, (time.perf_counter() - ready) * 1000


def _parse_start(message: dict) -> int:
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect()
    try:
        start = json.loads(message.get("text") or "")
    except json.JSONDecodeError:
        raise ProtocolError('first message must be JSON: {"type": "start", "sample_rate": <Hz>}') from None
    rate = start.get("sample_rate") if isinstance(start, dict) else None
    if not isinstance(start, dict) or start.get("type") != "start" or not isinstance(rate, int):
        raise ProtocolError('first message must be {"type": "start", "sample_rate": <Hz>}')
    if not SAMPLE_RATE_RANGE[0] <= rate <= SAMPLE_RATE_RANGE[1]:
        raise ProtocolError(f"sample_rate must be between {SAMPLE_RATE_RANGE[0]} and {SAMPLE_RATE_RANGE[1]} Hz")
    return rate


def _decode_chunk(data: bytes, sample_rate: int) -> np.ndarray:
    if len(data) % 4:
        raise ProtocolError("binary messages must be float32 samples (byte length divisible by 4)")
    if len(data) > MAX_CHUNK_SECONDS * sample_rate * 4:
        raise ProtocolError(f"chunks must be at most {MAX_CHUNK_SECONDS} s of audio")
    chunk = np.frombuffer(data, dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(chunk)):
        raise ProtocolError("audio contains NaN or infinity")
    return chunk


def _ndjson(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def create_app(engine: Engine | None = None, model_dir: str | Path | None = None,
               update_seconds: float | None = None) -> FastAPI:
    """Without `engine`, the model is loaded and timed at startup (about 10 s)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            path = model_dir or os.environ.get("ACCENT_MAP_MODEL") or DEFAULT_MODEL_DIR
            log.info("Loading model from %s and measuring inference time ...", path)
            app.state.engine = await asyncio.to_thread(build_engine, path, update_seconds)
        e = app.state.engine
        log.info("Ready: %s | one update every %.2f s (target %.2f s, measured %s ms per %.0f s window)",
                 e.model.config.display_name, e.settings.hop_seconds, e.model.config.audio.hop_seconds,
                 f"{e.measured_infer_ms:.0f}" if e.measured_infer_ms else "n/a", e.settings.window_seconds)
        yield
        app.state.engine.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Accent Map", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/api/model")
    def model_info() -> dict:
        e = app.state.engine
        c, s = e.model.config, e.settings
        return {
            "name": c.name,
            "display_name": c.display_name,
            "description": c.description,
            "caveats": c.caveats,
            "source": c.source.model_dump() if c.source else None,
            "classes": [cl.model_dump(exclude={"model_label"}) for cl in c.classes],
            "audio": {
                "sample_rate": e.model.sample_rate,
                "window_seconds": s.window_seconds,
                "target_update_seconds": c.audio.hop_seconds,
                "update_seconds": s.hop_seconds,
                "measured_infer_ms": e.measured_infer_ms,
            },
            "session": {k: getattr(s, k) for k in (
                "min_audio_seconds", "listen_speech_seconds", "tau_seconds",
                "confident_prob", "confident_margin", "top_k")},
        }

    @app.post("/api/analyze")
    async def analyze_file(file: UploadFile) -> StreamingResponse:
        e = app.state.engine
        sr = e.model.sample_rate
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // 2**20} MB).")
        try:
            audio = await asyncio.to_thread(decode_file, io.BytesIO(data), sr)
        except Exception as err:  # soundfile raises several types for unreadable input
            reason = getattr(err, "error_string", None) or type(err).__name__
            raise HTTPException(415, f"Could not read this audio file ({reason}). Supported: WAV, FLAC, OGG, MP3. "
                                     "Convert other formats (e.g. M4A) first.") from None
        duration = len(audio) / sr
        if duration > MAX_FILE_SECONDS:
            raise HTTPException(413, f"Audio too long: {duration:.0f} s (max {MAX_FILE_SECONDS} s).")
        if duration < e.settings.min_audio_seconds:
            raise HTTPException(422, f"Audio too short: {duration:.2f} s (need {e.settings.min_audio_seconds} s).")

        session = AnalysisSession(e.model, e.settings)
        hop = round(e.settings.hop_seconds * sr)

        async def lines():
            yield _ndjson({"type": "meta", "filename": file.filename, "duration": duration,
                           "update_seconds": e.settings.hop_seconds})
            count = 0
            for i in range(0, len(audio), hop):   # push one hop at a time to keep memory flat
                for pending in session.push(audio[i:i + hop]):
                    count += 1
                    yield _ndjson({"type": "event", **(await e.analyze(session, pending)).to_dict()})
            for pending in session.flush():
                count += 1
                yield _ndjson({"type": "event", **(await e.analyze(session, pending)).to_dict()})
            yield _ndjson({"type": "done", "events": count})

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.websocket("/ws/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        e = app.state.engine
        sender = None
        try:
            sample_rate = _parse_start(await ws.receive())
            stream = LiveStream(e, sample_rate)
            await ws.send_json({"type": "ready", "sample_rate": sample_rate,
                                "update_seconds": e.settings.hop_seconds})

            async def send_events():
                count = 0
                async for event, queue_ms, latency_ms in stream.events():
                    count += 1
                    await ws.send_json({"type": "event", **event.to_dict(), "dropped": stream.dropped,
                                        "queue_ms": queue_ms, "latency_ms": latency_ms})
                await ws.send_json({"type": "done", "events": count, "dropped": stream.dropped})

            sender = asyncio.create_task(send_events())
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if message.get("bytes") is not None:
                    stream.feed(_decode_chunk(message["bytes"], sample_rate))
                    continue
                try:
                    control = json.loads(message.get("text") or "")
                except json.JSONDecodeError:
                    raise ProtocolError("text messages must be JSON") from None
                if not isinstance(control, dict) or control.get("type") != "stop":
                    raise ProtocolError(f'expected {{"type": "stop"}}, got {message.get("text")!r}')
                stream.finish()
                break
            await sender
            await ws.close()
        except WebSocketDisconnect:
            if sender:
                sender.cancel()
        except ProtocolError as err:
            if sender:
                sender.cancel()
            await ws.send_json({"type": "error", "message": str(err)})
            await ws.close(code=1008)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()  # for: uvicorn --app-dir src server.app:app
