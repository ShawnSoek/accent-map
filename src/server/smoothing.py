"""Smooth class probabilities over time so the map moves instead of flickering.

Exponential moving average whose weight depends on the time since the last update:
alpha = 1 - exp(-dt / tau). After `tau` seconds a sudden change is 63 % visible,
whatever the update rate, so skipped or slower updates do not change the feel.
A long pause (dt >> tau) effectively restarts from the new prediction.
"""

from __future__ import annotations

import math

import numpy as np


class Smoother:
    def __init__(self, tau_seconds: float):
        if tau_seconds <= 0:
            raise ValueError("tau_seconds must be > 0")
        self.tau = tau_seconds
        self._value: np.ndarray | None = None
        self._t: float | None = None

    @property
    def value(self) -> np.ndarray | None:
        return None if self._value is None else self._value.copy()

    def update(self, probs: np.ndarray, t: float) -> np.ndarray:
        """Blend in a new probability vector observed at time t (seconds)."""
        probs = np.asarray(probs, dtype=np.float64)
        if self._value is None:
            self._value = probs.copy()
        else:
            if probs.shape != self._value.shape:
                raise ValueError(f"class count changed: {self._value.shape} -> {probs.shape}")
            dt = t - self._t
            if dt < 0:
                raise ValueError(f"time went backwards: {self._t} -> {t}")
            alpha = 1.0 - math.exp(-dt / self.tau)
            self._value = (1.0 - alpha) * self._value + alpha * probs
        self._value /= self._value.sum()  # keep it a distribution despite rounding
        self._t = t
        return self.value

    def reset(self) -> None:
        self._value = None
        self._t = None
