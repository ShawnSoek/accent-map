"""Random access to the analysis windows of a whole audio file.

In file mode the listener can jump around, so analysis must be able to start at any
position, not only at the beginning. FileAnalysis cuts the file once with the same
AnalysisSession.push logic as the live path, so window boundaries and "speech heard"
totals are identical, and it keeps only where each window ends. pendings_from(t) then
rebuilds the windows from the decoded audio, starting with the window that ends at or
just before t.

After a jump the smoother starts fresh (a new WindowAnalyzer per pass), but the speech
total still counts everything before the window, so there is no "listening" warm-up:
the 10 s before the playhead are in the file.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from server.model_api import AccentModel
from server.session import AnalysisSession, Pending, SessionSettings
from server.windowing import Window


@dataclass(frozen=True)
class _Mark:
    index: int
    end: int             # sample where the window ends (exclusive)
    new_from: int        # sample where its new audio starts (the previous window's end)
    speech_total: float  # sound heard from the start of the file up to `end`


class FileAnalysis:
    def __init__(self, model: AccentModel, settings: SessionSettings, audio: np.ndarray):
        self.sample_rate = model.sample_rate
        self.audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.window_samples = round(settings.window_seconds * self.sample_rate)
        hop = round(settings.hop_seconds * self.sample_rate)
        session = AnalysisSession(model, settings)
        self._marks: list[_Mark] = []
        for i in range(0, len(self.audio), hop):          # one hop at a time keeps memory flat
            self._record(session.push(self.audio[i:i + hop]))
        self._record(session.flush())
        self._ends = [m.end for m in self._marks]

    def _record(self, pendings: list[Pending]) -> None:
        for p in pendings:
            new_from = self._marks[-1].end if self._marks else 0
            self._marks.append(_Mark(p.window.index, p.window.end_sample, new_from, p.speech_total))

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def window_times(self) -> list[float]:
        return [end / self.sample_rate for end in self._ends]

    def start_index(self, t: float) -> int:
        """Index of the last window ending at or before t (0 if none does)."""
        i = bisect.bisect_right(self._ends, round(t * self.sample_rate)) - 1
        return max(i, 0)

    def pendings_from(self, t: float) -> Iterator[Pending]:
        for mark in self._marks[self.start_index(t):]:
            yield self._pending(mark)

    def _pending(self, m: _Mark) -> Pending:
        window = Window(
            index=m.index,
            end_sample=m.end,
            t=m.end / self.sample_rate,
            audio=self.audio[max(0, m.end - self.window_samples):m.end],
            new_audio=self.audio[m.new_from:m.end],
        )
        return Pending(window=window, speech_total=m.speech_total)
