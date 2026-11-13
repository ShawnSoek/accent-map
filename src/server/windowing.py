"""Cut a continuous audio stream into overlapping analysis windows.

A window ends at every hop boundary (k * hop, counted from the start of the stream)
and covers the last `window_seconds` of audio. At the start the window grows: the
first one is emitted at the first boundary with at least `min_seconds` of audio.
The same audio gives the same windows no matter how it is split into chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Window:
    index: int              # 0, 1, 2, ... in emission order
    end_sample: int         # absolute sample position where the window ends (exclusive)
    t: float                # end time in seconds since the start of the stream
    audio: np.ndarray       # the last min(window, end_sample) samples
    new_audio: np.ndarray   # samples since the previous window ended (or since the start)


class Windower:
    def __init__(self, sample_rate: int, window_seconds: float, hop_seconds: float, min_seconds: float):
        self.sample_rate = sample_rate
        self.window_samples = round(window_seconds * sample_rate)
        self.hop_samples = round(hop_seconds * sample_rate)
        self.min_samples = round(min_seconds * sample_rate)
        if not 0 < self.hop_samples <= self.window_samples:
            raise ValueError("need 0 < hop_seconds <= window_seconds")
        if not 0 < self.min_samples <= self.window_samples:
            raise ValueError("need 0 < min_seconds <= window_seconds")
        first_end = math.ceil(self.min_samples / self.hop_samples) * self.hop_samples
        if first_end > self.window_samples:
            raise ValueError("first window would end after window_seconds; lower min_seconds or hop_seconds")
        self._ring = np.zeros(self.window_samples, dtype=np.float32)
        self._pos = 0          # next write position in the ring
        self._total = 0        # samples received so far
        self._last_end = 0     # end_sample of the previous window
        self._next_end = first_end
        self._count = 0

    @property
    def total_samples(self) -> int:
        return self._total

    def push(self, chunk: np.ndarray) -> list[Window]:
        """Add audio; return the windows completed by it, oldest first."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"expected mono 1-D chunk, got shape {chunk.shape}")
        out = []
        pos = 0
        while pos < len(chunk):
            take = min(len(chunk) - pos, self._next_end - self._total)
            self._write(chunk[pos:pos + take])
            pos += take
            if self._total == self._next_end:
                out.append(self._emit())
                self._next_end += self.hop_samples
        return out

    def flush(self) -> Window | None:
        """At the end of a stream: a window ending at the last sample, if there is
        audio after the last window and enough audio overall. Otherwise None."""
        if self._total > self._last_end and self._total >= self.min_samples:
            return self._emit()
        return None

    def _write(self, data: np.ndarray) -> None:
        n, size = len(data), self.window_samples
        if n >= size:
            self._ring[:] = data[-size:]
            self._pos = 0
        else:
            first = min(n, size - self._pos)
            self._ring[self._pos:self._pos + first] = data[:first]
            self._ring[:n - first] = data[first:]
            self._pos = (self._pos + n) % size
        self._total += n

    def _latest(self, n: int) -> np.ndarray:
        """The last n samples (n <= window_samples), oldest first."""
        idx = np.arange(self._pos - n, self._pos) % self.window_samples
        return self._ring[idx]

    def _emit(self) -> Window:
        window = Window(
            index=self._count,
            end_sample=self._total,
            t=self._total / self.sample_rate,
            audio=self._latest(min(self._total, self.window_samples)),
            new_audio=self._latest(self._total - self._last_end),
        )
        self._count += 1
        self._last_end = self._total
        return window
