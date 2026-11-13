"""Pick the update interval the hardware can sustain.

The model config asks for an update every `hop_seconds` (0.5 s). If one inference
on a full window takes longer than that (plus headroom), the server updates less
often: the smallest multiple of `step` seconds that fits. On the dev laptop's CPU
a 5 s window takes ~0.8 s, so it updates every 1.0 s (DECISIONS.md, option 1).
"""

from __future__ import annotations

import math
import statistics
import time

import numpy as np

from server.model_api import AccentModel

HEADROOM = 1.2   # inference may use at most 1/1.2 = 83 % of the interval
STEP = 0.25      # seconds


def choose_update_interval(target_seconds: float, infer_ms: float,
                           headroom: float = HEADROOM, step: float = STEP) -> float:
    needed = infer_ms / 1000 * headroom
    if needed <= target_seconds:
        return target_seconds
    return math.ceil(round(needed / step, 9)) * step


def measure_inference_ms(model: AccentModel, window_seconds: float, runs: int = 3, warmup: int = 1) -> float:
    """Median time of model.predict on a full window of synthetic audio."""
    rng = np.random.default_rng(0)
    audio = (0.05 * rng.standard_normal(int(window_seconds * model.sample_rate))).astype(np.float32)
    times = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        model.predict(audio)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)
