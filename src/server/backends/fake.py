"""A deterministic stand-in model for tests and UI demos with any class count.

Class i is favoured when the audio's dominant frequency is near
base_hz + i * step_hz, so a sine tone selects a class. Near-silent audio
gives a uniform distribution. Options (backend_options in config.yaml):
  base_hz (default 200), step_hz (default 100), delay_seconds (default 0,
  sleeps per call to imitate a slow model).
"""

from __future__ import annotations

import time

import numpy as np

from server.model_api import AccentModel

SILENCE_RMS = 1e-4


class FakeAccentModel(AccentModel):
    def __init__(self, config, **_ignored):
        super().__init__(config)
        opts = config.backend_options
        unknown = set(opts) - {"base_hz", "step_hz", "delay_seconds"}
        if unknown:
            raise ValueError(f"unknown fake backend options: {sorted(unknown)}")
        self.base_hz = float(opts.get("base_hz", 200.0))
        self.step_hz = float(opts.get("step_hz", 100.0))
        self.delay_seconds = float(opts.get("delay_seconds", 0.0))

    def class_frequency(self, class_id: str) -> float:
        """The tone frequency (Hz) that selects this class."""
        return self.base_hz + self.class_ids.index(class_id) * self.step_hz

    def _predict(self, audio: np.ndarray) -> np.ndarray:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        n = len(self.class_ids)
        if np.sqrt(np.mean(audio.astype(np.float64) ** 2)) < SILENCE_RMS:
            return np.full(n, 1.0 / n)
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        peak = freqs[1 + int(np.argmax(spectrum[1:]))]  # skip DC
        centers = self.base_hz + self.step_hz * np.arange(n)
        logits = -2.0 * ((peak - centers) / self.step_hz) ** 2
        e = np.exp(logits - logits.max())
        return e / e.sum()
