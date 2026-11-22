"""Analysis sessions: audio in, a stream of display events out.

AnalysisSession serves the live (microphone) path: small chunks are pushed as they
arrive, and the server may skip windows when the model is busy. File mode
(server/file_analysis.py) cuts the file with the same AnalysisSession.push logic, so
both modes see identical windows, and both turn windows into events with
WindowAnalyzer.

Pipeline per window:
  energy gate -> model.predict -> Smoother -> status -> Event
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from server.audio import voiced_seconds
from server.model_api import MIN_AUDIO_SECONDS, AccentModel
from server.smoothing import Smoother
from server.windowing import Window, Windower

LISTENING = "listening"    # some speech heard, not enough yet to trust a result
NO_SPEECH = "no_speech"    # this window is (nearly) silent: no prediction, last result held
UNCERTAIN = "uncertain"    # enough speech, but no clear winner
CONFIDENT = "confident"


@dataclass(frozen=True)
class SessionSettings:
    """App behaviour, set by hand and documented in DECISIONS.md.
    Never tuned on data/own_tests/."""
    window_seconds: float = 5.0
    hop_seconds: float = 0.5               # the update interval actually used (see pacing.py)
    min_audio_seconds: float = 1.0         # first window needs this much audio
    gate_dbfs: float = -45.0               # 20 ms frames louder than this count as sound
    min_window_speech_seconds: float = 0.75  # less sound than this in a window -> NO_SPEECH
    listen_speech_seconds: float = 2.0     # LISTENING until this much sound was heard in total
    tau_seconds: float = 1.5               # smoothing time constant
    confident_prob: float = 0.5            # CONFIDENT needs top-1 >= this ...
    confident_margin: float = 0.15         # ... and top-1 minus top-2 >= this
    top_k: int = 3

    def __post_init__(self):
        if self.min_audio_seconds < MIN_AUDIO_SECONDS:
            raise ValueError(f"min_audio_seconds must be >= {MIN_AUDIO_SECONDS} (model contract)")


@dataclass(frozen=True)
class Pending:
    """A window waiting for analysis, with the speech total known when it was cut."""
    window: Window
    speech_total: float


@dataclass(frozen=True)
class Event:
    index: int
    t: float                          # end of the analyzed audio, seconds since start
    window_seconds: float             # length of the analyzed audio
    status: str
    raw: dict[str, float] | None      # this window's model output (None if not run)
    smoothed: dict[str, float] | None  # what the map shows (None before the first result)
    top: list[dict]                   # [{"id", "p"}], best first, from `smoothed`
    speech_seconds: float             # sound heard so far in the session
    infer_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def decide_status(smoothed: np.ndarray, speech_total: float, s: SessionSettings) -> str:
    if speech_total < s.listen_speech_seconds:
        return LISTENING
    top2 = np.sort(smoothed)[::-1][:2]
    if top2[0] >= s.confident_prob and top2[0] - top2[1] >= s.confident_margin:
        return CONFIDENT
    return UNCERTAIN


class WindowAnalyzer:
    """Window -> Event: energy gate, model, smoothing, status. Owns the smoother, so use one
    per stream of events (a live session, or one pass through a file from some position)."""

    def __init__(self, model: AccentModel, settings: SessionSettings):
        self.model = model
        self.settings = settings
        self.class_ids = model.class_ids
        self.smoother = Smoother(settings.tau_seconds)

    def analyze(self, pending: Pending) -> Event:
        s, w = self.settings, pending.window
        raw = infer_ms = None
        if voiced_seconds(w.audio, self.model.sample_rate, s.gate_dbfs) < s.min_window_speech_seconds:
            smoothed = self.smoother.value
            status = NO_SPEECH
        else:
            t0 = time.perf_counter()
            probs = self.model.predict(w.audio)
            infer_ms = (time.perf_counter() - t0) * 1000
            raw = np.array([probs[c] for c in self.class_ids])
            smoothed = self.smoother.update(raw, w.t)
            status = decide_status(smoothed, pending.speech_total, s)
        return Event(
            index=w.index,
            t=w.t,
            window_seconds=len(w.audio) / self.model.sample_rate,
            status=status,
            raw=self._as_dict(raw),
            smoothed=self._as_dict(smoothed),
            top=self._top(smoothed),
            speech_seconds=pending.speech_total,
            infer_ms=infer_ms,
        )

    def _as_dict(self, probs: np.ndarray | None) -> dict[str, float] | None:
        return None if probs is None else dict(zip(self.class_ids, probs.tolist()))

    def _top(self, probs: np.ndarray | None) -> list[dict]:
        if probs is None:
            return []
        order = np.argsort(-probs, kind="stable")[: self.settings.top_k]
        return [{"id": self.class_ids[i], "p": float(probs[i])} for i in order]


class AnalysisSession:
    """A stream of audio in arrival order (the live path): windows, speech totals, analysis."""

    def __init__(self, model: AccentModel, settings: SessionSettings):
        self.model = model
        self.settings = settings
        self.class_ids = model.class_ids
        self.windower = Windower(model.sample_rate, settings.window_seconds,
                                 settings.hop_seconds, settings.min_audio_seconds)
        self.analyzer = WindowAnalyzer(model, settings)
        self.speech_total = 0.0

    def push(self, audio: np.ndarray) -> list[Pending]:
        """Add audio at the model's sample rate; return windows ready for analysis."""
        return [self._pending(w) for w in self.windower.push(audio)]

    def flush(self) -> list[Pending]:
        """End of stream: a last window covering any audio after the final hop."""
        w = self.windower.flush()
        return [] if w is None else [self._pending(w)]

    def analyze(self, pending: Pending) -> Event:
        return self.analyzer.analyze(pending)

    def analyze_all(self, audio: np.ndarray):
        """Every window of `audio`, then the final partial one."""
        for pending in self.push(audio) + self.flush():
            yield self.analyze(pending)

    def _pending(self, w: Window) -> Pending:
        self.speech_total += voiced_seconds(w.new_audio, self.model.sample_rate, self.settings.gate_dbfs)
        return Pending(window=w, speech_total=self.speech_total)
